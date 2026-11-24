"""Plot saved cache-study CSVs. No simulator, model, or coding-agent dependency.

Run: python scripts/plot_cache_results.py --study result/agent-cache-budget-study/<run>
Requires matplotlib; input files are scores.csv and capacity.json.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

TRACES = ['sharegpt', 'mashqa', 'agentbank-apps', 'alfred', 'codeparrot']
LABELS = ['ShareGPT', 'MASH-QA', 'AgentBank APPS', 'ALFRED', 'CodeParrot']
POLICIES = ['LRU', 'LFU', 'FIFO', 'LeCaR', 'ARC', 'WA', 'UniCache', 'OPT',
            'OPT-Online', 'SimpleHYBRID-OPT', 'OFFSET-HEAD', 'OFFSET-TAIL', 'NONE', 'Agent']
SIMPLE = ['LRU', 'LFU', 'FIFO']
# Match the report's deterministic tie order. OPT and NONE are not online competitors.
ONLINE = ['UniCache', 'LRU', 'LFU', 'FIFO', 'LeCaR', 'ARC', 'WA',
          'OPT-Online', 'SimpleHYBRID-OPT', 'OFFSET-HEAD', 'OFFSET-TAIL']


def load(study, split):
    scores = {}
    with (study / 'scores.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            key = (int(row['gpu_blocks']), row['trace'], row['split'], row['policy'])
            if key in scores:
                raise ValueError(f'Duplicate score: {key}')
            value = float(row['hit_ratio']) if row['hit_ratio'] else None
            if value is not None and not (math.isfinite(value) and 0 <= value <= 1):
                raise ValueError(f'Invalid hit ratio: {key}')
            scores[key] = value
    capacity = json.loads((study / 'capacity.json').read_text())
    budgets = sorted({b for b, _, s, _ in scores if s == split})
    if not budgets:
        raise ValueError(f'No {split} scores')
    gib = {}
    for b in budgets:
        gib[b] = b * capacity['bytes_per_block'] / 2**30
        if not math.isclose(gib[b], capacity['budgets'][str(b)]['modeled_kv_gib'], abs_tol=1e-12):
            raise ValueError(f'Capacity metadata disagrees for {b}')
        for trace in TRACES:
            for policy in POLICIES[:-1]:
                if scores.get((b, trace, split, policy)) is None:
                    raise ValueError(f'Missing baseline: {b}/{trace}/{split}/{policy}')
    return scores, capacity, budgets, gib


def best_on_tune(scores, budget, trace, candidates):
    values = {p: scores.get((budget, trace, 'tune', p)) for p in candidates}
    if any(value is None for value in values.values()):
        raise ValueError(f'Missing tuning score: {budget}/{trace}')
    return max(candidates, key=lambda p: values[p])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='Default: STUDY/figures')
    parser.add_argument('--split', choices=['tune', 'eval'], default='eval')
    args = parser.parse_args()
    study = args.study.resolve()
    output = args.output.resolve() if args.output else study / 'figures'
    scores, capacity, budgets, gib = load(study, args.split)
    output.mkdir(parents=True, exist_ok=True)
    # Keep imports/cache independent of the repository and the user's home config.
    os.environ.setdefault('MPLCONFIGDIR', tempfile.mkdtemp(prefix='cache-figures-'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'savefig.facecolor': 'white'})

    def score(b, t, p):
        v = scores.get((b, t, args.split, p))
        return 100 * v if v is not None else math.nan

    def save(fig, name):
        for ext in ['png', 'pdf']:
            fig.savefig(output / f'{name}.{ext}', dpi=200)
        plt.close(fig)

    missing = [(gib[b], label) for b in budgets for trace, label in zip(TRACES, LABELS)
               if scores.get((b, trace, args.split, 'Agent')) is None]
    missing_note = ('Agent unavailable: ' + '; '.join(f'{t} at {g:g} GiB' for g, t in missing)
                    + '. Missing values are not plotted.') if missing else ''
    split_label = args.split.capitalize()
    display = lambda p: 'Offline OPT' if p == 'OPT' else p

    # Grouped bars: simple baselines only, with exact reported percentages.
    columns = min(2, len(budgets))
    rows = math.ceil(len(budgets) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(20, 5.2 * rows + 1), squeeze=False)
    width = .84 / len(SIMPLE)
    for ax, b in zip(axes.flat, budgets):
        for i, policy in enumerate(SIMPLE):
            xs = [j - .42 + width * (i + .5) for j in range(len(TRACES))]
            bars = ax.bar(xs, [score(b, t, policy) for t in TRACES], width=width,
                          label=display(policy), color=['#4c78a8', '#f28e2b', '#59a14f'][i],
                          edgecolor='white', linewidth=.5, zorder=3)
            ax.bar_label(bars, labels=[f'{bar.get_height():.2f}' for bar in bars],
                         padding=4, fontsize=9)
        ax.set_title(f'{gib[b]:g} GiB modeled KV capacity', fontsize=14)
        ax.set_xticks(range(len(TRACES)), LABELS)
        ax.set_xlabel('Workload')
        ax.set_ylabel(f'{split_label} token-weighted hit ratio (%)')
        ax.set_ylim(0, 106)
        ax.set_yticks(range(0, 101, 20))
        ax.grid(axis='y', alpha=.2, zorder=0)
    for ax in list(axes.flat)[len(budgets):]:
        ax.set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .035), ncol=3,
               frameon=False, fontsize=12)
    fig.suptitle('KV cache hit ratio by workload: simple baselines', fontsize=17, y=.985)
    fig.text(.5, .012, f'Bar labels are percentages: {args.split} token-weighted cache hit ratio. Each panel uses the same workload splits.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .115, 1, .96), h_pad=2)
    save(fig, 'hit-ratio-by-workload-policy')

    # One workload per panel; the same three policies as the grouped bars.
    summary = []
    fig, axes = plt.subplots(1, len(TRACES), figsize=(20, 5.8), squeeze=False)
    series = [('LRU', '#4c78a8', 'o', '-', 8),
              ('LFU', '#f28e2b', 's', '--', 6),
              ('FIFO', '#59a14f', '^', ':', 7)]
    for ax, trace, label in zip(axes[0], TRACES, LABELS):
        values = {name: [] for name, *_ in series}
        for b in budgets:
            for name, *_ in series:
                policy = name
                value = score(b, trace, policy)
                values[name].append(value)
                summary.append([b, gib[b], trace, args.split, name, policy,
                                '' if math.isnan(value) else value / 100])
        for name, color, marker, style, size in series:
            ax.plot([gib[b] for b in budgets], values[name], label=name, color=color,
                    marker=marker, markersize=size, linestyle=style, linewidth=1.5,
                    markerfacecolor='none' if marker in ['o', 's', 'D'] else color)
        ax.set_title(label, fontsize=13)
        ax.set_xlabel('Modeled KV capacity (GiB)')
        ax.set_xticks([gib[b] for b in budgets], [f'{gib[b]:.3f}'.rstrip('0').rstrip('.') for b in budgets])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.grid(alpha=.25)
    axes[0][0].set_ylabel(f'{split_label} token-weighted hit ratio (%)')
    handles, labels = axes[0][-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .065), ncol=3,
               frameon=False, fontsize=12)
    fig.suptitle('KV cache hit ratio versus modeled KV capacity: simple baselines', fontsize=17, y=.98)
    fig.text(.5, .025, 'Panels use different y-axis ranges. Overlapping curves indicate equal measured scores.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .17, 1, .93))
    save(fig, 'hit-ratio-by-modeled-kv-capacity')

    with (output / 'plotted-series.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['gpu_blocks', 'modeled_kv_gib', 'trace', 'split', 'series', 'policy', 'hit_ratio'])
        writer.writerows(summary)
    for source in [study / 'scores.csv', study / 'capacity.json', Path(__file__).resolve()]:
        target = output / source.name
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
    manifest = {'split': args.split, 'policies_in_bars': SIMPLE, 'policies_in_lines': SIMPLE,
                'capacity_formula': 'gpu_blocks * bytes_per_block / 2**30',
                'capacity_metadata': capacity, 'missing_agent_cases': missing,
                'python': sys.version, 'matplotlib': matplotlib.__version__,
                'inputs_sha256': {name: hashlib.sha256((study / name).read_bytes()).hexdigest()
                                  for name in ['scores.csv', 'capacity.json']}}
    (output / 'plot-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'README.md').write_text(
        '# Cache hit-ratio figures\n\n'
        'From this folder, reproduce both figures with Python and matplotlib:\n\n'
        f'```bash\npython plot_cache_results.py --study . --output . --split {args.split}\n```\n\n'
        'The copied CSV and capacity metadata are the complete plotting inputs. No simulation or API calls are needed. '
        'PNG and vector PDF versions are written. Input hashes are in plot-manifest.json.\n\n'
        'The grouped bars show LRU, LFU and FIFO at every capacity, with percentages above each bar and a common y-axis. '
        'The line figure shows the same three policies; panels have independent y-axis ranges. '
        'plotted-series.csv records every plotted point and its policy.\n\n'
        f'Modeled capacity uses {capacity["bytes_per_block"]:,} bytes per {capacity["block_tokens"]}-token '
        f'{capacity["kv_dtype"]} KV block and 2^30 bytes per GiB. It excludes model weights and other serving memory.\n\n'
        'Missing agent results stay blank; NONE is an actual zero-hit baseline. '
        'Offline OPT uses future information and is excluded from online baseline selection.\n\n'
        + missing_note + '\n')
    print(f'Figures and reproduction inputs: {output}')


if __name__ == '__main__':
    main()
