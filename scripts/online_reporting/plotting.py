"""Publication plots; no evidence loading or experiment execution."""
import statistics

import numpy as np

from export_online_results import arrow_axes
from .analysis import ARMS, TASKS

LABELS = ('Bridge-tower', 'Ring-on-peg', 'Tower relocation')
SHORT_LABELS = ('Bridge-tower', 'Ring', 'Relocation')
COLORS = {'Base': '#416A9C', 'Full': '#BD6833'}
FILLS = {'Base': '#D7E2F0', 'Full': '#F0DBCC'}


def save(fig, name, output):
    import matplotlib.pyplot as plt
    for suffix in ('pdf', 'png', 'svg'):
        metadata = {'CreationDate': None} if suffix == 'pdf' else {'Date': None} if suffix == 'svg' else {}
        fig.savefig(output / f'{name}.{suffix}', dpi=300, metadata=metadata)
        if suffix == 'svg':
            path = output / f'{name}.{suffix}'
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


def plot(episodes, summaries, pairs, curves, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9, 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'online-50'})
    bars = [Patch(facecolor=FILLS[a], edgecolor=COLORS[a], label=a) for a in ARMS]
    lines = [Line2D([], [], color=COLORS[a], lw=2, label=a) for a in ARMS]
    pooled = summaries['pooled_50']['arms']
    foot = '50 paired layouts per task. Audited repeats replace both arms of 9 timing/GPU-flagged pairs.'

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.95))
    fig.subplots_adjust(left=.065, right=.982, bottom=.25, top=.77, wspace=.36)
    for index, task in enumerate(TASKS):
        for arm, offset in [('Base', -.18), ('Full', .18)]:
            row = next(r for r in pooled if r['task'] == task and r['arm'] == arm)
            ys = [100 * row['mean_success'], row['mean_capped_seconds'], row['mean_total_tokens'] / 1000]
            labels = [f"{row['successes']}/50", f'{ys[1]:.1f}', f'{ys[2]:.1f}' + ('†' if not row['usage_complete'] else '')]
            for ax, y, label, pad in zip(axes, ys, labels, [2.5, 5, 6]):
                ax.bar(index + offset, y, width=.31, color=FILLS[arm], edgecolor=COLORS[arm], lw=1.1, zorder=2)
                ax.text(index + offset, y + pad, label, fontsize=7.8, ha='center', color=COLORS[arm])
    for ax, title, label, limit, ticks in zip(axes,
            ['(a) Success within 200 s', '(b) Capped completion time', '(c) Recorded tokens'],
            ['SR (%)', 'Time per episode (s)', 'Tokens per episode (k)'],
            [115, 235, 270], [range(0, 101, 20), range(0, 201, 50), range(0, 251, 50)]):
        arrow_axes(ax)
        ax.set_title(title, fontsize=11, pad=19)
        ax.set_ylabel(label, fontsize=10)
        ax.set(xticks=range(3), xticklabels=SHORT_LABELS, xlim=(-.55, 2.55), ylim=(0, limit), yticks=list(ticks))
    fig.legend(handles=bars, loc='upper center', ncol=2, frameon=False, fontsize=10)
    fig.text(.5, .085, foot, ha='center', fontsize=8.1, color='#4A4F55')
    fig.text(.5, .032, 'Time and tokens: episode means; timeouts = 200 s.  † Unknown interrupted tokens excluded.',
             ha='center', fontsize=8.1, color='#4A4F55')
    save(fig, 'outcome_bars', output)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.95))
    fig.subplots_adjust(left=.065, right=.978, bottom=.24, top=.77, wspace=.30)
    for index, (ax, task) in enumerate(zip(axes, TASKS)):
        for arm in ARMS:
            rows = [r for r in curves if r['task'] == task and r['arm'] == arm]
            ax.step([r['budget_seconds'] for r in rows], [r['sr_percent'] for r in rows],
                    where='post', color=COLORS[arm], lw=2)
        arrow_axes(ax)
        ax.set(xlabel='Elapsed wall time (s)', ylabel='SR (%)', xlim=(0, 203), ylim=(0, 105),
               xticks=[0, 50, 100, 150, 200], yticks=range(0, 101, 20))
        ax.set_title(f'({chr(97 + index)}) {LABELS[index]}', fontsize=11, pad=19)
    fig.legend(handles=lines, loc='upper center', ncol=2, frameon=False, fontsize=10)
    fig.text(.5, .075, foot, ha='center', fontsize=8.1, color='#4A4F55')
    fig.text(.5, .027, 'Steps mark first verified success in the same 200-second episodes; timeouts remain failures.',
             ha='center', fontsize=8.1, color='#4A4F55')
    save(fig, 'success_vs_time', output)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    fig.subplots_adjust(left=.075, right=.98, bottom=.25, top=.76, wspace=.29)
    for index, task in enumerate(TASKS):
        for arm, offset in [('Base', -.18), ('Full', .18)]:
            rows = sorted([r for r in episodes if r['task'] == task and r['arm'] == arm], key=lambda r: r['seed'])
            # Identical bounded offsets for both arms make paired layouts reproducible.
            jitter = np.random.default_rng(1050 + index).permutation(np.linspace(-.12, .12, len(rows)))
            for ax, key, scale in zip(axes, ['all_model_seconds', 'total_tokens'], [1, 1000]):
                ax.bar(index + offset, statistics.mean(r[key] / scale for r in rows), width=.31,
                       color=FILLS[arm], edgecolor=COLORS[arm], lw=.8, zorder=2)
                for j, row in enumerate(rows):
                    unknown = key == 'total_tokens' and not row['usage_complete']
                    ax.scatter(index + offset + jitter[j], row[key] / scale, s=18 if unknown else 12,
                               marker='^' if unknown else 'o', facecolors='white', edgecolors=COLORS[arm],
                               lw=.8, alpha=.85 if unknown else .65, zorder=4 if unknown else 3)
    for ax, title, ylabel in zip(axes, ['(a) Model-request time', '(b) Recorded token consumption'],
                                ['Model time per episode (s)', 'Tokens per episode (k)']):
        arrow_axes(ax)
        ax.set_title(title, fontsize=11, pad=19)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set(xticks=range(3), xticklabels=SHORT_LABELS, xlim=(-.55, 2.55), ylim=(0, ax.get_ylim()[1] * 1.04))
    handles = bars + [Line2D([], [], marker=m, ls='', markerfacecolor='white', color='.3', label=l)
                     for m, l in [('o', 'Episode'), ('^', 'Unknown extra tokens')]]
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False, fontsize=10)
    fig.text(.5, .080, 'Bars: mean of 50 episodes per arm/task. Points: selected episodes, including timeouts and 9 paired replacements.',
             ha='center', fontsize=8.1, color='#4A4F55')
    fig.text(.5, .028, 'Model time includes interrupted waits. Token totals exclude unknown interrupted usage.',
             ha='center', fontsize=8.1, color='#4A4F55')
    save(fig, 'model_resources', output)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.65))
    fig.subplots_adjust(left=.15, right=.97, bottom=.25, top=.76, wspace=.36)
    cohorts = [('pooled_50', 'Revised 50 pairs', '#BD6833', 'o', -.19),
               ('new_45', 'New 45 pairs', '#687585', 's', 0),
               ('sensitivity_valid_pairs_pooled', 'Quality sensitivity', '#267C70', 'D', .19)]
    for ax, metric, scale, title, xlabel in zip(axes,
            ['full_minus_base_sr', 'full_minus_base_seconds'], [100, 1],
            ['(a) Success difference', '(b) Capped-time difference'],
            ['Full − Base SR (percentage points)', 'Full − Base time (s)']):
        for cohort, label, color, marker, offset in cohorts:
            for i, task in enumerate(TASKS):
                row = next(r for r in summaries[cohort]['paired_differences'] if r['task'] == task and r['metric'] == metric)
                x, low, high = [row[k] * scale for k in ('mean', 'ci95_low', 'ci95_high')]
                ax.errorbar(x, i + offset, xerr=[[x - low], [high - x]], fmt=marker, color=color,
                            markersize=4.5, capsize=2.5, lw=1.15, label=label if i == 0 else None)
        ax.axvline(0, color='.5', ls=':', lw=.8)
        ax.set(yticks=range(3), yticklabels=LABELS, ylim=(2.55, -.55), xlabel=xlabel)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_color('.5')
        ax.tick_params(axis='y', length=0)
        ax.grid(axis='x', color='#E4E7EB', lw=.65)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=11, pad=15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, fontsize=9.5)
    fig.text(.5, .075, '95% paired-layout bootstrap intervals (10,000 resamples). Quality sensitivity excludes whole flagged pairs:',
             ha='center', fontsize=8.1, color='#4A4F55')
    fig.text(.5, .028, '48 Bridge-tower, 46 Ring, 47 Relocation pairs. Positive SR and negative time differences favor Full.',
             ha='center', fontsize=8.1, color='#4A4F55')
    save(fig, 'paired_effects', output)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    fig.subplots_adjust(left=.065, right=.98, bottom=.25, top=.78, wspace=.25)
    for index, (task, ax) in enumerate(zip(TASKS, axes)):
        selected = [p for p in pairs if p['task'] == task]
        selected.sort(key=lambda p: (p['cohort'] != 'original_5', p['seed']))
        for j, pair in enumerate(selected, 1):
            ax.plot([j - .12, j + .12], [pair['base_capped_seconds'], pair['full_capped_seconds']], color='.8', lw=.65)
            for arm, offset in [('Base', -.12), ('Full', .12)]:
                ax.scatter(j + offset, pair[f'{arm.lower()}_capped_seconds'], color=COLORS[arm],
                    marker='o' if pair[f'{arm.lower()}_success'] else 'x', s=12, linewidths=.7, zorder=3)
            if pair.get('replaced_pair', False):
                ax.scatter(j, 207, marker='v', color='.3', s=13)
        ax.axhline(200, color='.5', ls=':', lw=.8)
        ax.axvline(5.5, color='.7', ls=':', lw=.7)
        arrow_axes(ax)
        ax.set(xlabel='Paired layout (original 5, then new 45)', xticks=[1, 10, 20, 30, 40, 50], xlim=(0, 51), ylim=(0, 216))
        ax.set_title(f'({chr(97 + index)}) {LABELS[index]}', fontsize=10, pad=17)
    axes[0].set_ylabel('Capped completion time (s)')
    fig.legend(handles=lines, loc='upper center', ncol=2, frameon=False)
    fig.text(.5, .055, 'Circle: success. Cross: timeout. Triangle above 200 s: both arms replaced by their audited repeat.',
             ha='center', fontsize=8.1, color='#4A4F55')
    save(fig, 'paired_completion_times', output)
