import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
import pandas as pd
import numpy as np
import random
import scipy.stats
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
import torch
import torch.nn as nn
import torch.optim as optim
from joblib import Parallel, delayed
import json

# Attempt to import cuML models
try:
    from cuml.ensemble import RandomForestRegressor as cuRF
    from cuml.svm import SVR as cuSVR
    from cuml.linear_model import Ridge as cuRidge
    from cuml.neighbors import KNeighborsRegressor as cuKNeighborsRegressor
    from numba import cuda
    cuml_available = True
    print("cuML is loaded. GPU-accelerated models will be used.")
except ImportError:
    cuml_available = False
    print("cuML is not installed. CPU-based scikit-learn models will be used.")

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

# Common configuration
dataDir = '/home/zyr/PharmaTFormer/data'
output_dir = "/home/zyr/PharmaTFormer/data/ML"
os.makedirs(output_dir, exist_ok=True)


# Define PyTorch MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size=100, output_size=1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class PyTorchMLPRegressor:
    def __init__(self, input_size, hidden_size=100, output_size=1, lr=0.001, epochs=100, batch_size=32, device='cuda:0'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device if torch.cuda.is_available() else 'cpu'
        self._init_model()

    def _init_model(self):
        """Initialize/reinitialize the model, optimizer, and criterion."""
        self.model = MLP(self.input_size, self.hidden_size, self.output_size).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, X_train, y_train):
        self._init_model()
        self.model.train()
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

    def predict(self, X_val):
        self.model.eval()
        with torch.no_grad():
            X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            predictions = self.model(X_val).cpu().numpy().flatten()
        return predictions


def compute_drug_fingerprint(drug, smiles_series, radius=2, fpSize=1024):
    if not hasattr(compute_drug_fingerprint, "morgan_gen"):
        compute_drug_fingerprint.morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fpSize)
    morgan_gen = compute_drug_fingerprint.morgan_gen
    smiles = smiles_series.loc[drug]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES sequence for drug: {drug}")
        fingerprint = np.zeros(fpSize, dtype=int)
    else:
        fp = morgan_gen.GetFingerprint(mol)
        fingerprint = np.zeros((fpSize,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, fingerprint)
    return drug, fingerprint


def data_prepare_ml(dataDir, response=('AUC',), seed=seed, fpSize=1024):
    """
    Prepare machine learning data WITHOUT scaling (scaling moves into CV loop).
    Returns raw log2-transformed gene features + drug fingerprints.
    """
    # Read gene expression data
    expr = pd.read_csv(f'{dataDir}/ccle_rnaseq-2.csv', index_col=0)
    expr.dropna(inplace=True)
    expr = expr.apply(lambda x: np.log2(x + 1))

    # Read drug response data
    rsp = pd.read_csv(f'{dataDir}/ccle/PanCancer_drug_response.csv')
    rsp.dropna(inplace=True)
    colName = ['Cell Line Name', 'Drug Name', 'TCGA Classification']
    colName.extend(list(response))
    rsp = rsp[colName].copy()

    # Read drug SMILES data
    smiles_df = pd.read_csv(f'{dataDir}/ccle/drug_smiles.csv')
    smiles_df.set_index('Drug Name', inplace=True)
    smiles_df.index = smiles_df.index.str.strip()

    valid_drugs = rsp['Drug Name'].isin(smiles_df.index)
    rsp = rsp[valid_drugs]

    combined_df = pd.merge(rsp, expr.T, left_on='Cell Line Name', right_index=True)
    combined_df = combined_df.drop_duplicates(subset=['Cell Line Name', 'Drug Name'])

    y = combined_df[list(response)].values.flatten()
    x_gene = combined_df.drop(columns=list(response) + ['Drug Name', 'Cell Line Name', 'TCGA Classification'])
    x_gene = x_gene.apply(pd.to_numeric, errors='coerce').fillna(0)

    # NOTE: No scaling here — scaling is done inside each CV fold

    # Precompute unique drug fingerprints
    unique_drugs = combined_df['Drug Name'].unique()
    smiles_series = smiles_df['SMILES']
    drug_fingerprints = dict(Parallel(n_jobs=2)(
        delayed(compute_drug_fingerprint)(drug, smiles_series, fpSize=fpSize) for drug in unique_drugs
    ))

    drug_features = combined_df['Drug Name'].map(drug_fingerprints).values.tolist()
    drug_features = np.array(drug_features)

    assert x_gene.shape[0] == len(drug_features) == len(y), "Data size mismatch"

    drug_names = combined_df['Drug Name'].reset_index(drop=True)
    n_gene_features = x_gene.shape[1]

    # Return gene and drug features separately concatenated, plus boundary index
    X = np.hstack((x_gene.values, drug_features))

    return X, y, drug_names, x_gene.columns, n_gene_features


def evaluate_per_drug_correlation(y_true, y_pred, drug_names):
    drug_correlations = []
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    for drug in drug_names.unique():
        drug_mask = (drug_names == drug)
        if drug_mask.sum() < 5:
            continue
        drug_true = y_true[drug_mask]
        drug_pred = y_pred[drug_mask]
        try:
            pearson_corr, _ = scipy.stats.pearsonr(drug_true, drug_pred)
            spearman_corr, _ = scipy.stats.spearmanr(drug_true, drug_pred)
            if not np.isnan(pearson_corr) and not np.isnan(spearman_corr):
                drug_correlations.append({
                    'drug': drug,
                    'pearson': float(pearson_corr),
                    'spearman': float(spearman_corr),
                    'n_samples': int(drug_mask.sum())
                })
        except:
            continue

    if len(drug_correlations) == 0:
        return {
            'avg_pearson': np.nan,
            'avg_spearman': np.nan,
            'per_drug_results': [],
            'n_drugs': 0
        }

    avg_pearson = np.mean([d['pearson'] for d in drug_correlations])
    avg_spearman = np.mean([d['spearman'] for d in drug_correlations])

    return {
        'avg_pearson': avg_pearson,
        'avg_spearman': avg_spearman,
        'per_drug_results': drug_correlations,
        'n_drugs': len(drug_correlations)
    }


def scale_features_in_fold(X_train, X_val, n_gene_features):
    """
    Fit StandardScaler + MinMaxScaler on training gene features only,
    then transform both train and val. Drug fingerprints (binary) are not scaled.
    """
    # Split gene vs drug features
    X_train_gene = X_train[:, :n_gene_features].copy()
    X_val_gene = X_val[:, :n_gene_features].copy()
    X_train_drug = X_train[:, n_gene_features:]
    X_val_drug = X_val[:, n_gene_features:]

    # StandardScaler: fit on train only
    ss = StandardScaler()
    X_train_gene = ss.fit_transform(X_train_gene)
    X_val_gene = ss.transform(X_val_gene)

    # MinMaxScaler: fit on train only
    mms = MinMaxScaler()
    X_train_gene = mms.fit_transform(X_train_gene)
    X_val_gene = mms.transform(X_val_gene)

    # Recombine
    X_train_scaled = np.hstack((X_train_gene, X_train_drug))
    X_val_scaled = np.hstack((X_val_gene, X_val_drug))

    return X_train_scaled, X_val_scaled


def train_and_evaluate_fold(model_name, X, y, drug_names, n_gene_features,
                            train_index, val_index, fold_num, total_folds, device_id='cuda:0'):
    if cuml_available:
        try:
            cuda.select_device(0)
        except cuda.CudaAPIError as e:
            print(f"[{model_name}] Error selecting device 0: {e}")
            return np.nan, np.nan, 0, []

    print(f"[{model_name}] Fold {fold_num}/{total_folds} on {device_id}")

    X_train_raw, X_val_raw = X[train_index], X[val_index]
    y_train, y_val = y[train_index].astype(np.float32), y[val_index].astype(np.float32)
    drug_names_val = drug_names.iloc[val_index]

    # === Fix: scale inside fold, fit on train only ===
    X_train, X_val = scale_features_in_fold(X_train_raw, X_val_raw, n_gene_features)
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)

    # Initialize model
    if cuml_available:
        if model_name == 'RandomForest':
            model = cuRF(n_estimators=50, n_streams=1)
        elif model_name == 'SVR':
            model = cuSVR()
        elif model_name == 'Ridge':
            model = cuRidge(alpha=1.0)
        elif model_name == 'KNN':
            model = cuKNeighborsRegressor(n_neighbors=3)
        elif model_name == 'MLP':
            model = PyTorchMLPRegressor(input_size=X.shape[1], hidden_size=100, output_size=1,
                                        lr=0.001, epochs=100, batch_size=32, device=device_id)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    else:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.svm import SVR
        from sklearn.linear_model import Ridge
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.neural_network import MLPRegressor

        if model_name == 'RandomForest':
            model = RandomForestRegressor(random_state=seed, n_estimators=50, n_jobs=1)
        elif model_name == 'SVR':
            model = SVR()
        elif model_name == 'Ridge':
            model = Ridge(alpha=1.0, random_state=seed)
        elif model_name == 'KNN':
            model = KNeighborsRegressor(n_neighbors=3, n_jobs=1)
        elif model_name == 'MLP':
            model = MLPRegressor(hidden_layer_sizes=(500,), max_iter=100, random_state=seed)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    # Train and predict
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
    except Exception as e:
        print(f"[{model_name}] Error during training or prediction: {e}")
        return np.nan, np.nan, 0, []

    # Per-drug correlation (core metric)
    correlation_results = evaluate_per_drug_correlation(y_val, y_pred, drug_names_val)

    print(f"[{model_name}] Fold {fold_num} - Per-Drug Avg Pearson: {correlation_results['avg_pearson']:.4f}, "
          f"Per-Drug Avg Spearman: {correlation_results['avg_spearman']:.4f}, "
          f"N_drugs: {correlation_results['n_drugs']}")

    return (correlation_results['avg_pearson'], correlation_results['avg_spearman'],
            correlation_results['n_drugs'], correlation_results['per_drug_results'])


def train_and_evaluate_ml_models(X, y, drug_names, n_gene_features, k=5):
    model_names = ['RandomForest', 'SVR', 'Ridge', 'KNN', 'MLP']
    results = {}
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    for model_name in model_names:
        print(f"\nTraining model: {model_name}")

        fold_results = Parallel(n_jobs=1)(
            delayed(train_and_evaluate_fold)(
                model_name, X, y, drug_names, n_gene_features,
                train_index, val_index, fold + 1, k, device_id='cuda:0'
            )
            for fold, (train_index, val_index) in enumerate(kf.split(X))
        )

        # Collect per-fold results
        fold_results_detailed = []
        avg_pearson_scores = []
        avg_spearman_scores = []

        for fold_idx, (avg_pearson, avg_spearman, n_drugs, per_drug_details) in enumerate(fold_results):
            avg_pearson_scores.append(avg_pearson)
            avg_spearman_scores.append(avg_spearman)

            fold_results_detailed.append({
                'fold': int(fold_idx + 1),
                'avg_pearson': float(avg_pearson) if not np.isnan(avg_pearson) else None,
                'avg_spearman': float(avg_spearman) if not np.isnan(avg_spearman) else None,
                'n_drugs': int(n_drugs),
                'per_drug_details': json.dumps(per_drug_details)
            })

        # Save detailed fold results
        detailed_df = pd.DataFrame(fold_results_detailed)
        detailed_csv_path = os.path.join(output_dir, f'{model_name}_detailed_fold_results.csv')
        detailed_df.to_csv(detailed_csv_path, index=False)
        print(f"Detailed fold results for {model_name} saved to {detailed_csv_path}")

        # Average across folds
        avg_perdrug_pearson = np.nanmean(avg_pearson_scores) if not np.all(np.isnan(avg_pearson_scores)) else np.nan
        avg_perdrug_spearman = np.nanmean(avg_spearman_scores) if not np.all(np.isnan(avg_spearman_scores)) else np.nan

        results[model_name] = {
            'avg_perdrug_pearson': avg_perdrug_pearson,
            'avg_perdrug_spearman': avg_perdrug_spearman
        }

        print(f"\nModel: {model_name}")
        print(f"Average Per-Drug Pearson: {avg_perdrug_pearson:.4f}, Average Per-Drug Spearman: {avg_perdrug_spearman:.4f}")

    # Save summary
    summary_results = []
    for model_name, info in results.items():
        summary_results.append({
            'Model': model_name,
            'Avg_PerDrug_Pearson': info['avg_perdrug_pearson'],
            'Avg_PerDrug_Spearman': info['avg_perdrug_spearman']
        })

    summary_df = pd.DataFrame(summary_results)
    summary_csv_path = os.path.join(output_dir, 'ml_models_summary_results.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSummary ML results saved to {summary_csv_path}")

    return results


def main_ml_models():
    print("Preparing data...")
    X, y, drug_names, gene_columns, n_gene_features = data_prepare_ml(dataDir, fpSize=1024)
    print(f"Data shape: X={X.shape}, y={y.shape}, n_gene_features={n_gene_features}")

    print("Starting model training and evaluation...")
    results = train_and_evaluate_ml_models(X, y, drug_names, n_gene_features, k=5)


if __name__ == "__main__":
    main_ml_models()