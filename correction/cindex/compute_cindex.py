import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

ENDPOINTS = [
    ('COAD', '5-Fluorouracil'), ('COAD', 'Oxaliplatin'),
    ('BLCA', 'Gemcitabine'), ('BLCA', 'Cisplatin'), ('LIHC', 'Sorafenib'),
]


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_predictions(path, score_column):
    raw = pd.read_csv(path, dtype={'sample': str})
    required = ['sample', score_column, 'OS', 'OS.time']
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError(f'{path}: missing columns {sorted(missing)}')
    if raw['sample'].isna().any() or raw['sample'].duplicated().any():
        raise ValueError(f'{path}: sample IDs must be nonmissing and unique')
    data = raw[required].rename(columns={score_column: 'prediction'}).copy()
    for column in ['prediction', 'OS', 'OS.time']:
        data[column] = pd.to_numeric(data[column], errors='coerce')
    valid = np.isfinite(data[['prediction', 'OS', 'OS.time']]).all(axis=1)
    valid &= data['OS.time'] > 0
    removed = data.loc[~valid, 'sample'].tolist()
    data = data.loc[valid].copy()
    if not data['OS'].isin([0, 1]).all():
        raise ValueError(f'{path}: OS must be 0 (censored) or 1 (death)')
    return data, removed


def pair_counts(data):
    """Independent Harrell comparable-pair count, matching lifelines ties."""
    concordant = tied = comparable = 0
    values = data[['OS.time', 'OS', 'prediction']].to_numpy()
    for i, (time_i, event_i, risk_i) in enumerate(values):
        for time_j, event_j, risk_j in values[i + 1:]:
            if time_i == time_j:
                if event_i == event_j:
                    continue
                earlier_i = bool(event_i)
            elif time_i < time_j:
                if not event_i:
                    continue
                earlier_i = True
            else:
                if not event_j:
                    continue
                earlier_i = False
            comparable += 1
            if risk_i == risk_j:
                tied += 1
            elif (risk_i > risk_j) == earlier_i:
                concordant += 1
    if not comparable:
        raise ValueError('No comparable survival pairs')
    return concordant, tied, comparable


def calculate(data):
    # Fixed direction: a larger predicted AUC/response score denotes higher risk.
    # lifelines expects a score increasing with survival, hence the minus sign.
    score = float(concordance_index(data['OS.time'], -data['prediction'], data['OS']))
    correct, ties, pairs = pair_counts(data)
    independent = (correct + 0.5 * ties) / pairs
    if not np.isclose(score, independent, atol=1e-14, rtol=0):
        raise AssertionError('Independent pair count disagrees with lifelines')
    return score, {'comparable_pairs': pairs, 'concordant_pairs': correct,
                   'tied_prediction_pairs': ties}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Calculate OS C-indices.')
    parser.add_argument('--manifest', type=Path,
                        help='Optional CSV overriding the default prediction paths and score columns')
    parser.add_argument('--input-root', type=Path,
                        help='Base for prediction paths; defaults to manifest directory or current directory')
    parser.add_argument('--seed', type=int, default=42, choices=[42],
                        help='Fine-tuning run identifier; C-index calculation itself is deterministic')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/cindex_seed42'))
    args = parser.parse_args(argv)
    root = (args.input_root or (args.manifest.parent if args.manifest else Path.cwd())).resolve()
    if args.manifest:
        manifest = pd.read_csv(args.manifest, keep_default_na=False)
    else:
        manifest = pd.DataFrame([
            {'cohort': cohort, 'drug': drug, 'seed': args.seed,
             'before_csv': f'pretraining/{cohort}_{drug}_pretraining.csv',
             'before_score': 'prediction',
             'after_csv': f'finetuned_seed{args.seed}/{cohort}_{drug}_predictions_finetune_ensemble.csv',
             'after_score': f'{drug}_prediction'}
            for cohort, drug in ENDPOINTS
        ])
    columns = {'cohort', 'drug', 'seed', 'before_csv', 'before_score', 'after_csv', 'after_score'}
    if not columns <= set(manifest.columns):
        raise ValueError(f'Manifest requires columns {sorted(columns)}')
    if not (pd.to_numeric(manifest.seed, errors='raise') == args.seed).all():
        raise ValueError('Every manifest row must refer to the requested fine-tuning seed')
    if manifest.duplicated(['cohort', 'drug']).any():
        raise ValueError('Duplicate endpoints in manifest')
    if set(zip(manifest.cohort, manifest.drug)) != set(ENDPOINTS):
        raise ValueError('Manifest must contain exactly the five specified cohort/drug endpoints')
    indexed = manifest.set_index(['cohort', 'drug'])
    summaries, provenance, patients = [], [], []
    for cohort, drug in ENDPOINTS:
        row = indexed.loc[(cohort, drug)]
        before_path = (root / row.before_csv).resolve()
        after_path = (root / row.after_csv).resolve()
        before, invalid_before = read_predictions(before_path, row.before_score)
        after, invalid_after = read_predictions(after_path, row.after_score)
        before = before.sort_values('sample').reset_index(drop=True)
        after = after.sort_values('sample').reset_index(drop=True)
        if not before['sample'].equals(after['sample']):
            raise ValueError(f'{cohort}/{drug}: analyzed patients differ before and after fine-tuning')
        if not np.array_equal(before[['OS', 'OS.time']].to_numpy(), after[['OS', 'OS.time']].to_numpy()):
            raise ValueError(f'{cohort}/{drug}: survival outcomes differ between stages')
        before_c, before_counts = calculate(before)
        after_c, after_counts = calculate(after)
        endpoint = f'{cohort} / {drug}'
        summaries.append({'endpoint': endpoint, 'Before_fine_tuning': before_c,
                          'After_fine_tuning': after_c, 'Change': after_c - before_c,
                          'Patients': len(before), 'OS_events': int(before.OS.sum()),
                          'Fine_tuning_seed': args.seed})
        paired = before.rename(columns={'prediction': 'Before_prediction'})
        paired['After_prediction'] = after.prediction.to_numpy()
        paired.insert(0, 'endpoint', endpoint)
        patients.append(paired)
        provenance.append({'endpoint': endpoint, 'before_csv': str(before_path),
                           'after_csv': str(after_path), 'before_sha256': file_hash(before_path),
                           'after_sha256': file_hash(after_path),
                           'invalid_before': invalid_before, 'invalid_after': invalid_after,
                           'before': before_counts, 'after': after_counts})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output_dir / 'cindex_summary.csv', index=False)
    pd.concat(patients, ignore_index=True).to_csv(args.output_dir / 'paired_predictions.csv', index=False)
    (args.output_dir / 'calculation_manifest.json').write_text(json.dumps({
        'fine_tuning_seed': args.seed, 'metric': 'Harrell OS C-index',
        'score_direction': 'higher predicted response = higher risk; lifelines receives -prediction',
        'endpoints': provenance,
    }, indent=2) + '\n')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
