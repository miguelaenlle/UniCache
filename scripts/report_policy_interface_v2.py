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
    ap.add_argument('--matrix-dir',default='frozen-f1b4309-matrix')
    ap.add_argument('--shadow-dir',default='frozen-f1b4309-shadow')
    args=ap.parse_args()
    matrix=Path(args.matrix_dir)
    if not matrix.is_absolute():matrix=args.root/matrix
    shadowdir=Path(args.shadow_dir)
    if not shadowdir.is_absolute():shadowdir=args.root/shadowdir
    primary=read_rows(matrix)
    initial=read_rows(args.root/'full-reference-matrix')
    shadow=read_rows(shadowdir)
    cases=['sharegpt','mashqa','agentbank-apps','alfred','codeparrot','mixed']
    cells={(r['policy'],r['case']):r for r in primary}
    manifest=json.loads((matrix/'manifest.json').read_text()) if (matrix/'manifest.json').exists() else {}
    commit=manifest.get('git_head','unknown')
    lines=['# Policy interface v2 verification','',f'Runtime revision: `{commit}`.', '',
           'This validates editable reference implementations, not LLM rediscovery or improved hit ratios.',
           'Every independent v2 replay uses the same host LRU route; only the standalone `policy.py` changes.',
           'Each policy owns its tracking, histories, numerical models, weights and eviction decisions.',
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
    lines += ['', 'Numbers in parentheses are matched eviction decisions; zero-eviction cases alone do not establish pressure coverage.','',
              '## Additional gates','', '| Gate | Passed | Recorded cases | Matched evictions |', '|---|---:|---:|---:|']
    for title,rows in [('Fixed-host / max-seqs 32',primary),('Development original-host / max-seqs 1 (partial)',initial),('Native-shadow mixed / max-seqs 4',shadow)]:
        lines.append(f"| {title} | {sum(r['status']=='ok' for r in rows)} | {len(rows)} | {sum(r.get('native_evictions',0) for r in rows if r['status']=='ok'):,} |")
    lines += ['', 'Shadow runs retain native tracker behavior and native decisions. An isolated copy receives only',
              'the public facts and must propose the same victim at every native decision. Independent runs',
              'also exercise the new policy-neutral tracker route and its feedback on subsequent requests.', '',
              '## Reproduce','', 'From the repository root, with the existing local datasets/model metadata and simulator environment:', '', '```bash',
              '.venv-sim312/bin/python -m unittest policy_agent.test_v2_policies -v',
              'bash scripts/test-policy-interface-v2.sh --sessions 64 --jobs 4 --max-num-seqs 32 --output result/policy-interface-v2/reproduce-fixed',
              'bash scripts/test-policy-interface-v2.sh --traces --policies UniCache ARC LeCaR WA --jobs 2 --shadow --max-num-seqs 4 --output result/policy-interface-v2/reproduce-shadow',
              'python3 scripts/report_policy_interface_v2.py --matrix-dir reproduce-fixed --shadow-dir reproduce-shadow', '```', '',
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
              'The reducer also explicitly preserves native tracker reset quirks, deferred release times and',
              'post-allocation ghost-miss notification. Numerical code uses the environment versions recorded',
              'in each manifest. Exact parity is scoped to the tested inputs/settings; additional workloads,',
              'offload configurations and seeds can be tested with the same scripts.']
    args.root.mkdir(parents=True,exist_ok=True)
    (args.root/'report.md').write_text('\n'.join(lines)+'\n')
    print(args.root/'report.md')

if __name__=='__main__': main()
