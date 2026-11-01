"""Run outside a parent Seatbelt sandbox: python -m unittest policy_agent.test_sandbox."""
from pathlib import Path
import tempfile
import unittest
from policy_agent.bridge import Remote, PolicyFailure, ControlledEvictor

class IsolationTests(unittest.TestCase):
    def test_policy_state_and_choice(self):
        source = Path('policy_agent/starter.py').read_text().replace('return None', 'return min(self.blocks)')
        remote = Remote(source)
        try:
            result = remote.call({'events':[{'kind':'add','block_id':12}, {'kind':'add','block_id':3},
                                              {'kind':'remove','block_id':3}], 'choose':True,'now':0})
            self.assertEqual(result,12)
        finally:
            remote.close()

    def test_hidden_data_denied(self):
        with tempfile.NamedTemporaryFile(dir='result', mode='w') as hidden:
            hidden.write('private trace sentinel'); hidden.flush()
            source = 'class Policy:\n def __init__(self): open('+repr(str(Path(hidden.name).resolve()))+').read()\n'
            with self.assertRaisesRegex(PolicyFailure,'PermissionError'):
                Remote(source)

    def test_network_denied(self):
        with self.assertRaisesRegex(PolicyFailure,'PermissionError'):
            Remote("import socket\nclass Policy:\n def __init__(self): socket.create_connection(('127.0.0.1', 80))\n")

    def test_invalid_eviction_rejected_by_host(self):
        class Base:
            def __init__(self):
                self.blocks = {}
            def add(self, block_id, content_hash, num_hashed_tokens, last_accessed):
                self.blocks[block_id] = content_hash
            def __contains__(self, block_id):
                return block_id in self.blocks
        source = Path('policy_agent/starter.py').read_text().replace('return None', 'return -1')
        controller = ControlledEvictor(Base(), source)
        # Supply a clock without importing the full inference runtime in this unit test.
        def flush(choose):
            result = controller.remote.call({'events':controller.events,'choose':choose,'now':0})
            controller.events = []
            return result
        controller.flush = flush
        try:
            controller.add(12, 99, 8, 0)
            with self.assertRaisesRegex(PolicyFailure, 'InvalidEvictionError'):
                controller.evict()
            self.assertIn(12, controller.base)
        finally:
            controller.remote.close()

    def test_policy_traceback(self):
        with self.assertRaisesRegex(PolicyFailure, 'policy.py'):
            Remote("class Policy:\n def __init__(self): raise ValueError('example policy error')\n")

if __name__ == '__main__':
    unittest.main()
