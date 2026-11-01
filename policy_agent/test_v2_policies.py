"""Fast public-event contract checks; end-to-end parity is a separate CLI."""
import unittest
from policy_agent.v2_policies import POLICIES, source_for_policy


def make(name, capacity=10):
    scope={}
    exec(source_for_policy(name),scope)
    return scope['Policy']({'capacity':capacity,'block_size':8})


def emit(p,kind,**kw):
    p.on_event(dict(kind=kind,time=kw.pop('time',1.0),**kw))


def create(p,i,generation=1):
    emit(p,'block_allocated',block_id=i,generation=generation)
    emit(p,'hash_binding',block_id=i,content_hash=str(i),depth=8*(i+1))


def access(p,seq,ids,now=1.,workload='ShareGPT-multi-turn_test',out=0):
    emit(p,'sequence_step',sequence_id=seq,block_ids=ids,time=now,now=now,
         is_prefill=True,workload_class=workload,output_tokens_len=out)
    emit(p,'sequence_release',sequence_id=seq,block_ids=ids,time=now)


def eligible(p,i):
    emit(p,'eligibility_changed',block_id=i,content_hash=str(i),depth=8*(i+1),eligible=True)


class V2PoliciesTests(unittest.TestCase):
    def test_every_source_is_standalone(self):
        for name in POLICIES:
            with self.subTest(name=name):
                source=source_for_policy(name)
                self.assertNotIn('from vllm',source)
                self.assertNotIn('import vllm',source)
                p=make(name)
                for i in range(3): create(p,i)
                access(p,0,[0,1,2])
                for i in range(3): eligible(p,i)
                decision=p.evict({'now':2.})
                self.assertIn(decision['block_id'],range(3))
                self.assertEqual(decision['generation'],1)

    def test_start_source_does_not_bundle_other_algorithms(self):
        source=source_for_policy('LRU')
        for forbidden in ('ARCEvictor','SimpleHYBRIDEvictor','LeCaREvictor','maxlen=500','_KnownRequests'):
            self.assertNotIn(forbidden,source)
        self.assertNotIn('class LRUEvictor',source_for_policy('ARC'))

    def test_no_offline_opt(self):
        with self.assertRaises(ValueError): source_for_policy('OPT')

    def test_recency_vs_fifo(self):
        for name,expected in [('LRU',1),('FIFO',0)]:
            p=make(name)
            for i in range(2): create(p,i)
            access(p,0,[0],now=1.)
            access(p,1,[1],now=2.)
            access(p,2,[0],now=3.)
            for i in range(2): eligible(p,i)
            self.assertEqual(p.evict({'now':4.})['block_id'],expected)

    def test_arc_ghost_adaptation(self):
        p=make('ARC',capacity=2)
        for i in range(2): create(p,i)
        access(p,0,[0,1])
        for i in range(2): eligible(p,i)
        victim=p.evict({'now':2.})['block_id']
        self.assertIn(str(victim),p.algorithm.B1)
        emit(p,'lookup_completed',content_hash=str(victim),hit=False,time=3.)
        self.assertEqual(p.algorithm.p,1)
        self.assertNotIn(str(victim),p.algorithm.B1)

    def test_specialization_preserves_value_returning_or(self):
        scope={}
        exec(source_for_policy('UniCache'),scope)
        scope['Policy']({'capacity':10,'block_size':8})
        block=scope['SimpleHYBRIDlockMetaData']('x',8,0.,1,scope['TaskType'].SINGLE_TURN,False)
        score=block.calculate_priority(2048,{'single':1.})
        self.assertAlmostEqual(score,1-8/2048)

    def test_unicache_evict_first(self):
        p=make('UniCache')
        create(p,0);create(p,1)
        access(p,0,[0],workload='ShareGPT-multi-turn_test')
        access(p,1,[1],workload='apps_test')
        eligible(p,0);eligible(p,1)
        self.assertEqual(p.evict({'now':2.})['block_id'],1)

    def test_frequency_and_generation(self):
        p=make('LFU')
        create(p,0,generation=7);create(p,1)
        access(p,0,[0,1]);access(p,1,[1])
        eligible(p,0);eligible(p,1)
        self.assertEqual(p.evict({'now':2.}),{'block_id':0,'generation':7})

    def test_accounting_is_derived_from_raw_hits(self):
        p=make('UniCache')
        emit(p,'prefill_accounted',task_category='text',cached_prompt_tokens=80,accounting_ordinal=1)
        emit(p,'prefill_accounted',task_category='single',cached_prompt_tokens=16,accounting_ordinal=2)
        self.assertEqual(p.algorithm.output_simulator.step_track,2)
        self.assertEqual(list(p.algorithm.output_simulator.tracked_hit_stats['text']),[76.])
        self.assertEqual(list(p.algorithm.output_simulator.tracked_hit_stats['single']),[16])

if __name__=='__main__': unittest.main()
