import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
import os
import gc
from lifelines import CoxPHFitter
from sklearn.metrics import roc_curve
from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE
import codecs
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
random.seed(seed)

# General configuration
dataDir = '/home/zyr/PharmaTFormer/data'
epochSize = 10
batch_size = 128
lr = 0.00001
output_dir = "/home/zyr/PharmaTFormer/data"
device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

class FeatureExtractor(nn.Module):
    def __init__(self, gene_input_size, gene_hidden_size, drug_hidden_size):
        super(FeatureExtractor, self).__init__()
        self.gene_fc1 = nn.Linear(gene_input_size, gene_hidden_size)
        self.gene_fc2 = nn.Linear(gene_hidden_size, gene_hidden_size)
        self.smiles_fc = nn.Linear(128, drug_hidden_size)

    def forward(self, gene_expr, smiles):
        gene_out = F.relu(self.gene_fc1(gene_expr))
        gene_out = F.relu(self.gene_fc2(gene_out))
        smiles_out = F.relu(self.smiles_fc(smiles))
        combined_features = torch.cat((gene_out, smiles_out), dim=1)
        return combined_features

class TransModel(nn.Module):
    def __init__(self, feature_dim, nhead, seq_len, dim_feedforward=2048, dropout=0.1, num_layers=3):
        super(TransModel, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Sequential(
            nn.Linear(seq_len * feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = torch.flatten(x, 1)
        return self.output(x)

class CombinedModel(nn.Module):
    def __init__(self, gene_input_size, gene_hidden_size, drug_hidden_size, feature_dim, nhead, num_layers=3, dim_feedforward=2048, dropout=0.1):
        super(CombinedModel, self).__init__()
        self.feature_extractor = FeatureExtractor(gene_input_size, gene_hidden_size, drug_hidden_size)
        self.feature_dim = feature_dim
        self.seq_len = (gene_hidden_size + drug_hidden_size) // feature_dim
        self.transformer = TransModel(
            feature_dim=feature_dim,
            nhead=nhead,
            seq_len=self.seq_len,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

    def forward(self, gene_expr, smiles):
        features = self.feature_extractor(gene_expr, smiles)
        batch_size = features.size(0)
        features = features.view(batch_size, self.seq_len, self.feature_dim)
        output = self.transformer(features)
        return output

def prepare_data(x_gene, x_drug, y, batch_size=batch_size, seed=seed):
    def worker_init_fn(worker_id):
        np.random.seed(seed)
        random.seed(seed)

    x_gene_tensor = torch.tensor(x_gene.values).float()
    x_drug_tensor = torch.tensor(x_drug).float()
    y_tensor = torch.tensor(y.values).float()

    # Debugging: Print tensor shapes
    print(f"x_gene_tensor shape: {x_gene_tensor.shape}")
    print(f"x_drug_tensor shape: {x_drug_tensor.shape}")
    print(f"y_tensor shape: {y_tensor.shape}")

    # Check for size mismatch
    assert x_gene_tensor.shape[0] == x_drug_tensor.shape[0] == y_tensor.shape[0], "Size mismatch between tensors"

    dataset = torch.utils.data.TensorDataset(x_gene_tensor, x_drug_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=worker_init_fn
    )
    return dataloader

def data_prepare(dataDir, response=('AUC',), seed=seed):
    expr = pd.read_csv(f'{dataDir}/ccle_rnaseq-2.csv', index_col=0)
    expr.dropna(inplace=True)
    expr = expr.apply(lambda x: np.log2(x + 1))

    rsp = pd.read_csv(f'{dataDir}/ccle/PanCancer_drug_response.csv')
    rsp.dropna(inplace=True)
    colName = ['Cell Line Name', 'Drug Name']
    colName.extend(list(response))
    rsp = rsp[colName].copy()

    smiles_df = pd.read_csv(f'{dataDir}/ccle/drug_smiles.csv')
    smiles_df.set_index('Drug Name', inplace=True)
    smiles_df.index = smiles_df.index.str.strip()

    valid_drugs = rsp['Drug Name'].isin(smiles_df.index)
    rsp = rsp[valid_drugs]

    combined_df = pd.merge(rsp, expr.T, left_on='Cell Line Name', right_index=True)
    combined_df = combined_df.drop_duplicates(subset=['Cell Line Name', 'Drug Name'])

    y = combined_df[list(response)]
    x_gene = combined_df.drop(columns=list(response) + ['Drug Name', 'Cell Line Name'])

    x_gene = x_gene.apply(pd.to_numeric, errors='coerce').fillna(0)

    bpe_codes_path = f'{dataDir}/bpe.codes'
    with codecs.open(f'{dataDir}/ccle/drug_smiles.csv', encoding='utf-8') as f_in:
        with codecs.open(bpe_codes_path, 'w', encoding='utf-8') as f_out:
            learn_bpe(f_in, f_out, num_symbols=10000)

    with codecs.open(bpe_codes_path, encoding='utf-8') as f_in:
        bpe = BPE(f_in)

    smiles_encoded = []

    for idx, row in combined_df.iterrows():
        drug = row['Drug Name']
        encoded_smile = [ord(char) for char in bpe.process_line(smiles_df.loc[drug, 'SMILES'])]
        if len(encoded_smile) > 128:
            encoded_smile = encoded_smile[:128]
        padded_smile = np.pad(encoded_smile, (0, 128 - len(encoded_smile)), 'constant')
        smiles_encoded.append(padded_smile)

    smiles_encoded = np.array(smiles_encoded)

    assert x_gene.shape[0] == len(smiles_encoded) == y.shape[0], "Data sizes are not aligned"

    return x_gene, smiles_encoded, y

def save_ensemble_model(models, save_path):
    ensemble_state_dict = {f'model_{i + 1}': model.state_dict() for i, model in enumerate(models)}
    torch.save(ensemble_state_dict, save_path)
    print(f"Ensemble model saved to {save_path}")

def train_and_evaluate_kfold(x_gene, x_smiles, y, k=5, epochs=10, lr=lr, batch_size=batch_size):
    # Standardization and Normalization
    standard_scaler = StandardScaler()
    x_gene_standardized = pd.DataFrame(standard_scaler.fit_transform(x_gene), columns=x_gene.columns)

    min_max_scaler = MinMaxScaler()
    x_gene_normalized = pd.DataFrame(min_max_scaler.fit_transform(x_gene_standardized), columns=x_gene.columns)

    x_gene = x_gene_normalized  # Now x_gene is standardized and normalized

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    models = []
    val_losses = []
    pearson_coeffs = []
    spearman_coeffs = []

    for fold, (train_index, val_index) in enumerate(kf.split(x_gene)):
        print(f"Fold {fold + 1}/{k}")
        
        x_gene_train, x_gene_val = x_gene.iloc[train_index], x_gene.iloc[val_index]
        x_smiles_train, x_smiles_val = x_smiles[train_index], x_smiles[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model = CombinedModel(
            gene_input_size=x_gene_train.shape[1],
            gene_hidden_size=8192,
            drug_hidden_size=256,
            feature_dim=128,
            nhead=8
        )
        model = model.to(device)
        torch.cuda.empty_cache()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_loader = prepare_data(x_gene_train, x_smiles_train, y_train, batch_size, seed=seed)
        val_loader = prepare_data(x_gene_val, x_smiles_val, y_val, batch_size, seed=seed)

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for gene_inputs, smiles_inputs, targets in train_loader:
                gene_inputs, smiles_inputs, targets = gene_inputs.to(device), smiles_inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(gene_inputs, smiles_inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            print(f'Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}')

        model.eval()
        val_loss = 0.0
        all_outputs = []
        all_targets = []
        with torch.no_grad():
            for gene_inputs, smiles_inputs, targets in val_loader:
                gene_inputs, smiles_inputs, targets = gene_inputs.to(device), smiles_inputs.to(device), targets.to(device)
                outputs = model(gene_inputs, smiles_inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                all_outputs.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())

        val_loss /= len(val_loader)
        # Compute Pearson and Spearman correlations
        pearson_corr = np.corrcoef(all_outputs, all_targets)[0, 1]
        spearman_corr, _ = spearmanr(all_outputs, all_targets)
        print(f'Validation Loss (Fold {fold + 1}): {val_loss:.4f}')
        print(f'Pearson Correlation (Fold {fold + 1}): {pearson_corr:.4f}')
        print(f'Spearman Correlation (Fold {fold + 1}): {spearman_corr:.4f}')

        models.append(model)
        val_losses.append(val_loss)
        pearson_coeffs.append(pearson_corr)
        spearman_coeffs.append(spearman_corr)

    avg_val_loss = np.mean(val_losses)
    avg_pearson_corr = np.mean(pearson_coeffs)
    avg_spearman_corr = np.mean(spearman_coeffs)
    print(f'Average Validation Loss: {avg_val_loss:.4f}')
    print(f'Average Pearson Correlation: {avg_pearson_corr:.4f}')
    print(f'Average Spearman Correlation: {avg_spearman_corr:.4f}')

    # Save ensemble model
    ensemble_save_path = os.path.join(output_dir, "ensemble_model-1.pth")
    save_ensemble_model(models, ensemble_save_path)

    return models, avg_val_loss

def load_ensemble_model(save_path, gene_input_size, gene_hidden_size, drug_hidden_size, feature_dim, nhead):
    ensemble_state_dict = torch.load(save_path)
    models = []

    for i in range(len(ensemble_state_dict)):
        model = CombinedModel(
            gene_input_size=gene_input_size,
            gene_hidden_size=gene_hidden_size,
            drug_hidden_size=drug_hidden_size,
            feature_dim=feature_dim,
            nhead=nhead
        )
        model.load_state_dict(ensemble_state_dict[f'model_{i + 1}'])
        model = model.to(device)
        models.append(model)
    
    return models

def predict_with_ensemble(models, gene_expr, smiles, batch_size=16):
    predictions = []

    for model in models:
        model.eval()
        gene_expr = gene_expr.apply(pd.to_numeric, errors='coerce').fillna(0)
        data_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(gene_expr.values).float(),
                torch.tensor(smiles).float()
            ),
            batch_size=batch_size,
            shuffle=False
        )

        fold_predictions = []
        with torch.no_grad():
            for gene_inputs, smiles_inputs in data_loader:
                gene_inputs, smiles_inputs = gene_inputs.to(device), smiles_inputs.to(device)
                outputs = model(gene_inputs, smiles_inputs)
                fold_predictions.extend(outputs.cpu().numpy().flatten())

        predictions.append(np.array(fold_predictions))

    ensemble_predictions = np.mean(predictions, axis=0)
    return ensemble_predictions

def process_tcga_data(data_path, cancer_type='BLCA', drug_name='Gemcitabine', output_dir='/home/zyr/PharmaTFormer/data', standardization=True, normalization=True):
    expr = pd.read_csv(data_path, index_col=0, header=0)
    expr.dropna(inplace=True)
    expr = expr.apply(lambda x: np.log2(x + 1))
    expr = expr.T

    expr = expr.apply(pd.to_numeric, errors='coerce').fillna(0)

    if standardization:
        scaler = StandardScaler()
        expr = pd.DataFrame(scaler.fit_transform(expr), columns=expr.columns, index=expr.index)

    if normalization:
        scaler = MinMaxScaler()
        expr = pd.DataFrame(scaler.fit_transform(expr), columns=expr.columns, index=expr.index)

    smiles_df = pd.read_csv(f'{dataDir}/ccle/drug_smiles.csv')
    smiles_df.set_index('Drug Name', inplace=True)
    smiles_df.index = smiles_df.index.str.strip()

    if drug_name not in smiles_df.index:
        raise ValueError(f"Drug {drug_name} not found in SMILES dataset.")

    bpe_codes_path = f'{dataDir}/bpe.codes'
    with codecs.open(bpe_codes_path, encoding='utf-8') as f_in:
        bpe = BPE(f_in)

    encoded_smile = [ord(char) for char in bpe.process_line(smiles_df.loc[drug_name, 'SMILES'])]
    if len(encoded_smile) > 128:
        encoded_smile = encoded_smile[:128]
    smiles_encoded = np.pad(encoded_smile, (0, 128 - len(encoded_smile)), 'constant')
    smiles_encoded = np.tile(smiles_encoded, (expr.shape[0], 1))

    return expr, smiles_encoded

def load_clinical_data(drug, cancer_type='BLCA'):
    try:
        clinical_data_path = f'/home/zyr/PharmaTFormer/data/merge/TCGA_{cancer_type}_merged_{drug}_survival_1.csv'
        clinical_data = pd.read_csv(clinical_data_path)
        clinical_data.columns = clinical_data.columns.str.strip()
        clinical_data = clinical_data[['sample', 'OS', 'OS.time', 'PFI.time', 'PFI']].dropna()
        if 'sample' not in clinical_data.columns:
            raise ValueError(f"'sample' column is missing in the clinical data for drug: {drug}")

        return clinical_data
    except ValueError as e:
        print(e)
        return pd.DataFrame()

def evaluate_on_clinical_data(models, cancer_type, drug, tcga_data_path):
    tcga_expr_processed, smiles_encoded = process_tcga_data(
        tcga_data_path,
        cancer_type=cancer_type,
        drug_name=drug
    )
    predictions = predict_with_ensemble(models, tcga_expr_processed, smiles_encoded)

    result_df = pd.DataFrame(predictions, index=tcga_expr_processed.index, columns=[f'{drug}_prediction'])
    result_save_path = os.path.join(output_dir, f"{cancer_type}_{drug}_predictions_ensemble-1.csv")
    result_df.to_csv(result_save_path)
    print(f"Ensemble Predictions saved to {result_save_path}")

    clinical_data = load_clinical_data(drug, cancer_type=cancer_type)
    if clinical_data.empty:
        print(f"No clinical data available for drug: {drug} in cancer type: {cancer_type}. Skipping evaluation.")
        return

    merged_data = result_df.merge(clinical_data, left_index=True, right_on='sample')

    # Ensure required columns exist
    required_columns = ['OS', 'OS.time', 'PFI', 'PFI.time']
    missing_columns = [col for col in required_columns if col not in merged_data.columns]
    if missing_columns:
        print(f"Missing columns {missing_columns} in clinical data for drug: {drug}. Skipping evaluation.")
        return

    try:
        fpr, tpr, thresholds = roc_curve(merged_data['OS'], merged_data[f'{drug}_prediction'])
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        merged_data['risk_group'] = (merged_data[f'{drug}_prediction'] > optimal_threshold).astype(int)
    except Exception as e:
        print(f"Error computing ROC curve for drug: {drug}. Error: {e}")
        merged_data['risk_group'] = 0  

    output_columns = [f'{drug}_prediction', 'risk_group', 'OS', 'OS.time', 'PFI', 'PFI.time']
    output_df = merged_data[output_columns]
    detailed_result_save_path = os.path.join(output_dir, f"{cancer_type}_{drug}_detailed_predictions-pretrain.csv")
    output_df.to_csv(detailed_result_save_path, index=False)
    print(f"Detailed Predictions with Risk Groups and Clinical Data saved to {detailed_result_save_path}")

    cph = CoxPHFitter()
    try:
        cph.fit(merged_data[['OS.time', 'OS', 'risk_group']], duration_col='OS.time', event_col='OS')
        hr = np.exp(cph.summary['coef'].values[0])
        hr_ci_lower = np.exp(cph.summary['coef lower 95%'].values[0])
        hr_ci_upper = np.exp(cph.summary['coef upper 95%'].values[0])
        p_value = cph.summary['p'].values[0]

        print(f"Drug: {drug}")
        print(f"Optimal threshold: {optimal_threshold}")
        print(f"Hazard Ratio: {hr} (95% CI: {hr_ci_lower}-{hr_ci_upper})")
        print(f"P-value: {p_value}\n")
    except Exception as e:
        print(f"Error fitting CoxPH model for drug: {drug}. Error: {e}")

    plot_survival_curves(merged_data, drug, cancer_type)

def plot_survival_curves(data, drug, cancer_type):
    kmf = KaplanMeierFitter()

    survival_plot_dir = os.path.join(output_dir, "survival_plots")
    os.makedirs(survival_plot_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    for label, group in data.groupby('risk_group'):
        kmf.fit(durations=group['OS.time'], event_observed=group['OS'], label=f'Risk Group {label}')
        kmf.plot_survival_function()
    plt.title(f'Overall Survival (OS) Kaplan-Meier Curve for {drug} in {cancer_type}')
    plt.xlabel('Time (days)')
    plt.ylabel('Survival Probability')
    plt.legend()
    os_plot_path = os.path.join(survival_plot_dir, f"{cancer_type}_{drug}_OS_KM_curve.png")
    plt.savefig(os_plot_path)
    plt.close()
    print(f"OS Kaplan-Meier curve saved to {os_plot_path}")

    plt.figure(figsize=(8, 6))
    for label, group in data.groupby('risk_group'):
        kmf.fit(durations=group['PFI.time'], event_observed=group['PFI'], label=f'Risk Group {label}')
        kmf.plot_survival_function()
    plt.title(f'Progression-Free Interval (PFI) Kaplan-Meier Curve for {drug} in {cancer_type}')
    plt.xlabel('Time (days)')
    plt.ylabel('Survival Probability')
    plt.legend()
    pfi_plot_path = os.path.join(survival_plot_dir, f"{cancer_type}_{drug}_PFI_KM_curve.png")
    plt.savefig(pfi_plot_path)
    plt.close()
    print(f"PFI Kaplan-Meier curve saved to {pfi_plot_path}")

def main_kfold():
    x_gene, x_smiles, y = data_prepare(dataDir)
    models, avg_val_loss = train_and_evaluate_kfold(x_gene, x_smiles, y, k=5, epochs=epochSize, batch_size=batch_size)

    tcga_data_paths = {
        'BLCA': f'{dataDir}/TCGA-BLCA-gene-adjust-2.csv',
        'COAD': f'{dataDir}/TCGA-COAD-gene-adjust-2.csv'
    }
    drugs = {
        'BLCA': ["Gemcitabine", "Cisplatin"],
        'COAD': ["5-Fluorouracil", "Oxaliplatin"]
    }

    for cancer_type, data_path in tcga_data_paths.items():
        for drug in drugs[cancer_type]:
            evaluate_on_clinical_data(models, cancer_type, drug, data_path)

if __name__ == "__main__":
    main_kfold()
