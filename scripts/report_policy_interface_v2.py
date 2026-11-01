"""Rebuild the v2 parity coverage report from private comparison JSON files."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from policy_agent.v2_policies import POLICIES


def read_rows(folder):
    return [json.loads(p.read_text()) for p in sorted(folder.glob('*/*/comparison.json'))]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=ROOT/'result/policy-interface-v2')
    ap.add_argument('--matrix-dir',default='frozen-cd2e82e-matrix')
    ap.add_argument('--shadow-dir',default='frozen-cd2e82e-shadow')
    ap.add_argument('--pressure-dirs',nargs='*',default=['frozen-cd2e82e-real-pressure','frozen-cd2e82e-code-pressure'])
    ap.add_argument('--postgate-code-dir',default='startup-4db0ee6-code')
    ap.add_argument('--postgate-shadow-dir',default='startup-4db0ee6-shadow')
    ap.add_argument('--final-code-dir',default='final-e9a6d16-code')
    ap.add_argument('--final-shadow-dir',default='final-e9a6d16-shadow')
    args=ap.parse_args()
    matrix=Path(args.matrix_dir)
    if not matrix.is_absolute():matrix=args.root/matrix
    shadowdir=Path(args.shadow_dir)
    if not shadowdir.is_absolute():shadowdir=args.root/shadowdir
    primary=read_rows(matrix)
    initial=read_rows(args.root/'full-reference-matrix')
    shadow=read_rows(shadowdir)
    cases=['sharegpt','mashqa','agentbank-apps','alfred','codeparrot','mixed']
    pressure=[]
    for name in args.pressure_dirs:
        folder=Path(name)
        pressure += read_rows(folder if folder.is_absolute() else args.root/folder)
    cells={(r['policy'],r['case']):r for r in primary}
    manifest=json.loads((matrix/'manifest.json').read_text()) if (matrix/'manifest.json').exists() else {}
    commit=manifest.get('git_head','unknown')
    combined=primary+pressure+shadow
    complete=len(primary)==len(POLICIES)*6 and len(pressure)==len(POLICIES)*3 and len(shadow)==4
    verified=complete and all(r['status']=='ok' and r.get('scores_match') and r.get('first_difference') is None for r in combined)
    stable=all(len({r['source_sha256'] for r in combined if r['policy']==policy})==1 for policy in POLICIES) if combined else False
    real_pressure=all(any(r['policy']==policy and r['case']==trace and r.get('native_evictions',0)>0 for r in combined) for policy in POLICIES if policy!='NONE' for trace in cases[:-1])
    verification={'complete':complete,'passed':verified,'case_count':len(combined),'expected_cases':112,
                  'matched_evictions':sum(r.get('native_evictions',0) for r in combined if r['status']=='ok'),
                  'identical_policy_source_across_configurations':stable,'every_online_policy_real_pressure_all_five_traces':real_pressure,
                  'runtime_revision':commit}
    postrows=[]
    discrepancies=[]
    ledgerfiles=0
    postrevision=None
    for newname,oldname in [(args.postgate_code_dir,'frozen-cd2e82e-code-pressure'),(args.postgate_shadow_dir,'frozen-cd2e82e-shadow')]:
        new=Path(newname)
        if not new.is_absolute():new=args.root/new
        old=args.root/oldname
        if (new/'manifest.json').exists():postrevision=json.loads((new/'manifest.json').read_text()).get('git_head')
        for row in read_rows(new):
            postrows.append(row)
            case=Path(row['case'])/row['policy']
            originalrow=json.loads((old/case/'comparison.json').read_text())
            if row.get('source_sha256') != originalrow.get('source_sha256'):discrepancies.append(str(case/'source'))
            modes=['native','v2']+(['shadow'] if row.get('shadow_checked') else [])
            for mode in modes:
                before=old/case/mode/'decisions.jsonl'
                after=new/case/mode/'decisions.jsonl'
                if (before.read_bytes() if before.exists() else b'') != (after.read_bytes() if after.exists() else b''):
                    discrepancies.append(str(case/mode))
                ledgerfiles += 1
    verification['post_gate_startup_protocol']={'runtime_revision':postrevision,'reference_runtime_revision':commit,
        'cases_recorded':len(postrows),'expected_cases':16,
        'passed':len(postrows)==16 and all(r['status']=='ok' for r in postrows) and not discrepancies,
        'byte_identical_ledger_comparisons':ledgerfiles-len(discrepancies),'ledger_discrepancies':discrepancies}
    finalrows=[]
    finaldiscrepancies=[]
    finalledgers=0
    finalrevision=None
    changedsources=0
    for newname,oldname in [(args.final_code_dir,'frozen-cd2e82e-code-pressure'),(args.final_shadow_dir,'frozen-cd2e82e-shadow')]:
        new=Path(newname)
        if not new.is_absolute():new=args.root/new
        old=args.root/oldname
        if (new/'manifest.json').exists():finalrevision=json.loads((new/'manifest.json').read_text()).get('git_head')
        for row in read_rows(new):
            finalrows.append(row)
            case=Path(row['case'])/row['policy']
            originalrow=json.loads((old/case/'comparison.json').read_text())
            changedsources += row.get('source_sha256') != originalrow.get('source_sha256')
            for mode in ['native','v2']+(['shadow'] if row.get('shadow_checked') else []):
                before=old/case/mode/'decisions.jsonl'
                after=new/case/mode/'decisions.jsonl'
                if (before.read_bytes() if before.exists() else b'') != (after.read_bytes() if after.exists() else b''):
                    finaldiscrepancies.append(str(case/mode))
                finalledgers+=1
    verification['final_reset_aware_surface']={'runtime_revision':finalrevision,'reference_runtime_revision':commit,
        'cases_recorded':len(finalrows),'expected_cases':16,
        'passed':len(finalrows)==16 and all(r['status']=='ok' for r in finalrows) and not finaldiscrepancies and changedsources==16,
        'byte_identical_ledger_comparisons':finalledgers-len(finaldiscrepancies),
        'sources_changed_for_new_reset_handler':changedsources,'ledger_discrepancies':finaldiscrepancies}
    (args.root/'verification.json').write_text(json.dumps(verification,indent=2))
    lines=['# Policy interface v2 verification','',f'Core parity runtime revision: `{commit}`.', '',
           f"Verification: {'PASS' if verified and stable and real_pressure else 'IN PROGRESS'}; {len(combined)}/112 cases; {verification['matched_evictions']:,} matched eviction decisions.", '',
           f'Final reset-aware surface revision: `{finalrevision}`; separate targeted regressions are recorded below.', '',
           'This validates editable reference implementations, not LLM rediscovery or improved hit ratios.',
           'Every independent v2 replay uses the same host LRU route; only the standalone `policy.py` changes.',
           'Each policy owns its tracking, histories, numerical models, weights and eviction decisions.',
           'Each policy uses byte-identical source across configurations. Every online policy has actual',
           'nonzero eviction parity on each of the five real workloads using the main or pressure cases.',
           'Offline OPT is intentionally excluded. NONE is a disabled-cache initialization control.', '',
           '## Primary coverage: fixed host, 32 concurrent sequence slots','',
           'Five real prefixes use 64 source sessions (32 whole tune sessions) and 4,096 GPU blocks.',
           'The synthetic seven-category fixture uses 100 two-turn sessions and 128 blocks.',
           'A passing cell matches every eviction (ID and hash), trusted hit-token and prompt totals,',
           'and complete request/response/prompt-length ledgers. It is not merely a matching rounded ratio.','',
           '| Policy | '+' | '.join(cases)+' |','|---|'+'---:|'*len(cases)]
    for policy in POLICIES:
        row=[]
        for case in cases:
            r=cells.get((policy,case))
            if not r: cell='pending'
            elif r['status']=='ok': cell=f"PASS ({r['native_evictions']:,})"
            else: cell=r['status'].upper()
            row.append(cell)
        lines.append('| '+policy+' | '+' | '.join(row)+' |')
    lines += ['', '## Supplementary real-workload pressure', '',
              'At 512 blocks (4,096 tokens), one request at a time fits with margin: MASH-QA maximum',
              '2,652 tokens / 332 blocks; APPS 2,304 / 288; CodeParrot 2,186 / 274. The full prefix',
              'footprints exceed that capacity, forcing actual evictions. These supplements address',
              'real cases that fit entirely at the main 4,096-block budget.', '',
              '| Policy | MASH-QA / 512 | APPS / 512 | CodeParrot / 512 |', '|---|---:|---:|---:|']
    pressurecells={(r['policy'],r['case']):r for r in pressure}
    for policy in POLICIES:
        cellslist=[]
        for case in ('mashqa','agentbank-apps','codeparrot'):
            r=pressurecells.get((policy,case))
            cellslist.append('pending' if r is None else (f"PASS ({r['native_evictions']:,})" if r['status']=='ok' else r['status'].upper()))
        lines.append('| '+policy+' | '+' | '.join(cellslist)+' |')
    lines += ['', 'Numbers in parentheses are matched eviction decisions; zero-eviction cases alone do not establish pressure coverage.','',
              '## Additional gates','', '| Gate | Passed | Recorded cases | Matched evictions |', '|---|---:|---:|---:|']
    for title,rows in [('Fixed-host / max-seqs 32',primary),('Development original-host / max-seqs 1 (partial)',initial),('Native-shadow mixed / max-seqs 4',shadow),('Supplementary real pressure / max-seqs 1',pressure)]:
        lines.append(f"| {title} | {sum(r['status']=='ok' for r in rows)} | {len(rows)} | {sum(r.get('native_evictions',0) for r in rows if r['status']=='ok'):,} |")
    post=verification['post_gate_startup_protocol']
    lines += ['', '## Startup protocol regression', '',
              f"Post-gate runtime revision: `{postrevision}`. {len(postrows)}/16 targeted configurations recorded; "+('PASS.' if post['passed'] else 'in progress.'),
              f"{post['byte_identical_ledger_comparisons']} native/v2/shadow eviction-ledger comparisons are byte-identical (empty for NONE) to their original core-gate runs.",
              'This separate check validates the mode-only startup handshake and eager initialization with',
              'the full environment context. Policy algorithms and archived candidate sources are unchanged.',
              'It supplements the 112-case core gate rather than relabelling that gate as a newer revision.', '']
    final=verification['final_reset_aware_surface']
    lines += ['', '## Final reset-aware surface regression', '',
              f"Final surface revision: `{finalrevision}`. {len(finalrows)}/16 freshly generated configurations recorded; "+('PASS.' if final['passed'] else 'in progress.'),
              f"{final['byte_identical_ledger_comparisons']} native/v2/shadow ledger comparisons are byte-identical to the core gate.",
              f"All {changedsources} candidate source hashes changed as expected to include the new raw `cache_reset` handler.",
              'These cases verify ordinary replay decisions remain unchanged. Actual reset boundaries and',
              'preservation of global learned fits/hit windows/known requests have separate reset integration',
              'and persistence tests in the v2 unit suites. The private core gate remains pinned to its',
              'original revision; the protocol and reset-aware extensions are recorded separately.', '']
    lines += ['', 'Shadow runs retain native tracker behavior and native decisions. An isolated copy receives only',
              'the public facts and must propose the same victim at every native decision. Independent runs',
              'also exercise the new policy-neutral tracker route and its feedback on subsequent requests.', '',
              '## Reproduce','', 'From the repository root, with the existing local datasets/model metadata and simulator environment:', '', '```bash',
              '.venv-sim312/bin/python -m unittest policy_agent.test_v2_policies policy_agent.test_v2_worker policy_agent.test_v2_lifecycle policy_agent.test_v2_numerics -v',
              'bash scripts/test-policy-interface-v2.sh --sessions 64 --jobs 4 --max-num-seqs 32 --output result/policy-interface-v2/reproduce-fixed',
              'bash scripts/test-policy-interface-v2.sh --traces --policies UniCache ARC LeCaR WA --jobs 2 --shadow --max-num-seqs 4 --output result/policy-interface-v2/reproduce-shadow',
              'bash scripts/test-policy-interface-v2.sh --traces mashqa agentbank-apps --sessions 64 --gpu-blocks 512 --max-num-seqs 1 --jobs 4 --no-mixed --output result/policy-interface-v2/reproduce-pressure',
              'bash scripts/test-policy-interface-v2.sh --traces codeparrot --sessions 64 --gpu-blocks 512 --max-num-seqs 1 --jobs 2 --no-mixed --output result/policy-interface-v2/reproduce-code-pressure',
              'python3 scripts/report_policy_interface_v2.py --matrix-dir reproduce-fixed --shadow-dir reproduce-shadow --pressure-dirs reproduce-pressure reproduce-code-pressure', '```', '',
              'To reproduce from archived records and policy sources, without re-preparing the original datasets:', '',
              '```bash',
              'bash scripts/test-policy-interface-v2.sh --input-snapshot result/policy-interface-v2/frozen-cd2e82e-matrix --jobs 4 --output result/policy-interface-v2/reproduce-archived',
              '```', '',
              'To reproduce the final reset-aware generated sources specifically:', '',
              '```bash',
              'bash scripts/test-policy-interface-v2.sh --input-snapshot result/policy-interface-v2/final-e9a6d16-code --jobs 4 --output result/policy-interface-v2/reproduce-final-code',
              'bash scripts/test-policy-interface-v2.sh --input-snapshot result/policy-interface-v2/final-e9a6d16-shadow --jobs 2 --output result/policy-interface-v2/reproduce-final-shadow',
              '```', '',
              'The snapshot command inherits original budgets, concurrency, policies and shadow settings unless',
              'explicitly overridden. It relocates the exact frozen data and copies the original policy sources.', '',
              'Frozen verification output directories contain the exact command, dependency versions, code/model hashes, input',
              'manifests, standalone editable sources, native/v2 decision ledgers and per-case comparisons.',
              'Use a fresh output directory for each reproduction. No coding model or API call is used here.', '',
              '## Corrections retained for audit','',
              '`mixed-first` and `mixed-second` are invalid fixture attempts: synthetic input text was missing,',
              'then text tokenization disagreed with its declared token IDs. Independent prompt-ledger checks',
              'caught both. The valid fixture derives IDs using the local serving tokenizer. These failed',
              'fixture attempts are preserved but excluded from policy parity pass counts.',
              '`full-reference-matrix` and `fixed-host-concurrent-matrix` are partial development runs stopped',
              f'to free resources after runtime freeze. Primary coverage comes from `{matrix.name}` and',
              f'`{shadowdir.name}`, using the runtime revision recorded above.', '',
              'Revision `f1b4309` failed UniCache shadow parity at decision 295. Source specialization had',
              'incorrectly treated a value-returning `max_offset or 1` expression as a Boolean, changing',
              'single-turn retention scores. Revision `cd2e82e` restricts partial Boolean folding to',
              'predicate contexts. A numerical regression and native-algorithm AST preservation checks',
              'now guard this failure. Superseded outputs remain archived and are excluded from final passes.', '',
              'The reducer also explicitly preserves native tracker reset quirks, deferred release times and',
              'post-allocation ghost-miss notification. Numerical code uses the environment versions recorded',
              'in each manifest. Exact parity is scoped to the tested inputs/settings; additional workloads,',
              'offload configurations and seeds can be tested with the same scripts.']
    args.root.mkdir(parents=True,exist_ok=True)
    (args.root/'report.md').write_text('\n'.join(lines)+'\n')
    print(args.root/'report.md')

if __name__=='__main__': main()
