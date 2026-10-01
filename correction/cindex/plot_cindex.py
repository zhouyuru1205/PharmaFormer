import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ENDPOINTS = ['COAD / 5-Fluorouracil', 'COAD / Oxaliplatin',
             'BLCA / Gemcitabine', 'BLCA / Cisplatin', 'LIHC / Sorafenib']


def main(argv=None):
    parser = argparse.ArgumentParser(description='Plot OS C-indices.')
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/cindex_seed42'))
    args = parser.parse_args(argv)
    data = pd.read_csv(args.summary, float_precision='round_trip')
    if data.endpoint.duplicated().any() or set(data.endpoint) != set(ENDPOINTS):
        raise ValueError('Expected one row for each of the five cohort/drug endpoints')
    data = data.set_index('endpoint').loc[ENDPOINTS].reset_index()
    if not (data.Fine_tuning_seed == 42).all():
        raise ValueError('This figure expects the seed-42 fine-tuning analysis')
    before = data.Before_fine_tuning.to_numpy()
    after = data.After_fine_tuning.to_numpy()
    values = np.concatenate([before, after])
    if not np.isfinite(values).all() or (values < 0).any() or (values > .8).any():
        raise ValueError('Values are invalid or exceed the requested 0–0.8 plotting range')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.labelsize': 12, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#9CAAB4', 'text.color': '#233341', 'axes.labelcolor': '#233341',
        'xtick.color': '#233341', 'ytick.color': '#53626E',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    colors = ['#B8C3CD', '#087F8C']
    x, width = np.arange(len(data)), .31
    fig, ax = plt.subplots(figsize=(11.6, 6.2))
    fig.subplots_adjust(left=.085, right=.98, bottom=.14, top=.76)
    bars_before = ax.bar(x - width / 2, before, width, color=colors[0],
                         label='Before fine-tuning', zorder=3)
    bars_after = ax.bar(x + width / 2, after, width, color=colors[1],
                        label='After fine-tuning', zorder=3)
    for bars, is_after in [(bars_before, False), (bars_after, True)]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .013,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=11,
                    color=colors[1] if is_after else '#53626E',
                    fontweight='bold' if is_after else 'normal')
    ax.set_xticks(x, [name.replace(' / ', '\n') for name in ENDPOINTS])
    ax.tick_params(axis='x', length=0, pad=12)
    ax.set_ylim(0, .8)
    ax.set_yticks(np.arange(0, .81, .2))
    ax.set_ylabel('OS C-index')
    ax.grid(axis='y', color='#E8EDF0', lw=.8, zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(.5, color='#89969F', lw=.8, ls=(0, (4, 4)), zorder=1)
    fig.text(.085, .94, 'Survival discrimination before and after fine-tuning',
             fontsize=17, fontweight='bold')
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.005), frameon=False,
              ncol=2, borderaxespad=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(args.output_dir / f'Cindex_before_after_finetuning.{extension}',
                    dpi=300, facecolor='white')
    plt.close(fig)
    (args.output_dir / 'figure_manifest.json').write_text(json.dumps({
        'summary_sha256': hashlib.sha256(args.summary.read_bytes()).hexdigest(),
        'fine_tuning_seed': 42, 'ylim': [0, .8], 'subtitle': None,
        'endpoints': ENDPOINTS, 'before': before.tolist(), 'after': after.tolist(),
        'patient_counts': data.Patients.astype(int).tolist(),
    }, indent=2) + '\n')
    print(args.output_dir / 'Cindex_before_after_finetuning.png')


if __name__ == '__main__':
    main()
