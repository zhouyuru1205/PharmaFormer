import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
import os
import gc
from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE
import codecs
from scipy.stats import spearmanr

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
random.seed(seed)

# General configuration
dataDir = '/home/zyr/PharmaTFormer/data'
epochSize = 100
batch_size = 128
lr = 0.00001
output_dir = "/home/zyr/PharmaTFormer/data"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

def prepare_data(x_gene, x_drug, y, batch_size=batch_size, seed=seed, shuffle=True):
    def worker_init_fn(worker_id):
        np.random.seed(seed)
        random.seed(seed)

    x_gene_tensor = torch.tensor(x_gene.values).float()
    x_drug_tensor = torch.tensor(x_drug).float()
    y_tensor = torch.tensor(y.values).float()

    print(f"x_gene_tensor shape: {x_gene_tensor.shape}")
    print(f"x_drug_tensor shape: {x_drug_tensor.shape}")
    print(f"y_tensor shape: {y_tensor.shape}")

    assert x_gene_tensor.shape[0] == x_drug_tensor.shape[0] == y_tensor.shape[0], "Size mismatch between tensors"

    dataset = torch.utils.data.TensorDataset(x_gene_tensor, x_drug_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        worker_init_fn=worker_init_fn
    )
    return dataloader

def data_prepare(dataDir, response=('AUC',), seed=seed):
    expr = pd.read_csv(f'{dataDir}/ccle_rnaseq-2.csv', index_col=0)
    expr.dropna(inplace=True)
    expr = expr.apply(lambda x: np.log2(x + 1))

    rsp = pd.read_csv(f'{dataDir}/ccle/PanCancer_drug_response.csv')
    rsp.dropna(inplace=True)
    colName = ['Cell Line Name', 'Drug Name', 'TCGA Classification']
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
    x_gene = combined_df.drop(columns=list(response) + ['Drug Name', 'Cell Line Name', 'TCGA Classification'])
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

    drug_names = combined_df['Drug Name'].reset_index(drop=True)
    cell_lines = combined_df['Cell Line Name'].reset_index(drop=True)
    tcga_types = combined_df['TCGA Classification'].reset_index(drop=True)

    return x_gene, smiles_encoded, y, drug_names, cell_lines, tcga_types

def evaluate_per_drug_correlation(all_outputs, all_targets, drug_names):
    drug_correlations = []
    all_outputs = np.array(all_outputs)
    all_targets = np.array(all_targets)
    
    for drug in drug_names.unique():
        drug_mask = (drug_names == drug)
        if drug_mask.sum() < 5:
            continue
            
        drug_outputs = all_outputs[drug_mask]
        drug_targets = all_targets[drug_mask]
        
        try:
            pearson_corr = np.corrcoef(drug_outputs, drug_targets)[0, 1]
            spearman_corr, _ = spearmanr(drug_outputs, drug_targets)
            
            if not np.isnan(pearson_corr) and not np.isnan(spearman_corr):
                drug_correlations.append({
                    'drug': drug,
                    'pearson': pearson_corr,
                    'spearman': spearman_corr,
                    'n_samples': drug_mask.sum()
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

def train_and_evaluate_kfold(x_gene, x_smiles, y, drug_names, cell_lines, tcga_types, k=5, epochs=10, lr=lr, batch_size=batch_size):
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    models = []
    val_losses = []
    fold_results = []
    
    print(f"Total samples: {len(x_gene)}")
    print(f"Unique cell lines: {len(cell_lines.unique())}")
    print(f"Unique drugs: {len(drug_names.unique())}")
    print(f"Using KFold with {k} splits")

    for fold, (train_index, val_index) in enumerate(kf.split(x_gene)):
        print(f"Fold {fold + 1}/{k}")
        
        x_gene_train, x_gene_val = x_gene.iloc[train_index], x_gene.iloc[val_index]
        x_smiles_train, x_smiles_val = x_smiles[train_index], x_smiles[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        drug_names_val = drug_names.iloc[val_index]
        cell_lines_train = cell_lines.iloc[train_index]
        cell_lines_val = cell_lines.iloc[val_index]
        print(f"Training set: {len(train_index)} samples, {len(cell_lines_train.unique())} unique cell lines")
        print(f"Validation set: {len(val_index)} samples, {len(cell_lines_val.unique())} unique cell lines")
        
        train_cell_set = set(cell_lines_train.unique())
        val_cell_set = set(cell_lines_val.unique())
        overlap = train_cell_set.intersection(val_cell_set)
        print(f"Cell line overlap between train and validation: {len(overlap)} cell lines")

        standard_scaler = StandardScaler()
        x_gene_train_standardized = pd.DataFrame(
            standard_scaler.fit_transform(x_gene_train), 
            columns=x_gene_train.columns,
            index=x_gene_train.index
        )
        x_gene_val_standardized = pd.DataFrame(
            standard_scaler.transform(x_gene_val), 
            columns=x_gene_val.columns,
            index=x_gene_val.index
        )

        min_max_scaler = MinMaxScaler()
        x_gene_train_normalized = pd.DataFrame(
            min_max_scaler.fit_transform(x_gene_train_standardized), 
            columns=x_gene_train_standardized.columns,
            index=x_gene_train_standardized.index
        )
        x_gene_val_normalized = pd.DataFrame(
            min_max_scaler.transform(x_gene_val_standardized), 
            columns=x_gene_val_standardized.columns,
            index=x_gene_val_standardized.index
        )

        x_gene_train = x_gene_train_normalized
        x_gene_val = x_gene_val_normalized

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

        train_loader = prepare_data(x_gene_train, x_smiles_train, y_train, batch_size, seed=seed, shuffle=True)
        val_loader = prepare_data(x_gene_val, x_smiles_val, y_val, batch_size, seed=seed, shuffle=False)

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
        
        correlation_results = evaluate_per_drug_correlation(all_outputs, all_targets, drug_names_val)
        
        print(f'Validation Loss (Fold {fold + 1}): {val_loss:.4f}')
        print(f'Per-Drug Avg Pearson Correlation (Fold {fold + 1}): {correlation_results["avg_pearson"]:.4f}')
        print(f'Per-Drug Avg Spearman Correlation (Fold {fold + 1}): {correlation_results["avg_spearman"]:.4f}')
        print(f'Number of drugs in validation: {correlation_results["n_drugs"]}')

        models.append(model)
        val_losses.append(val_loss)
        fold_results.append({
            'fold': fold + 1,
            'val_loss': val_loss,
            'avg_pearson': correlation_results['avg_pearson'],
            'avg_spearman': correlation_results['avg_spearman'],
            'n_drugs': correlation_results['n_drugs'],
            'per_drug_details': correlation_results['per_drug_results'],
            'train_cell_lines': len(cell_lines_train.unique()),
            'val_cell_lines': len(cell_lines_val.unique()),
            'train_samples': len(train_index),
            'val_samples': len(val_index)
        })

    avg_val_loss = np.mean(val_losses)
    avg_perdrug_pearson = np.mean([r['avg_pearson'] for r in fold_results if not np.isnan(r['avg_pearson'])])
    avg_perdrug_spearman = np.mean([r['avg_spearman'] for r in fold_results if not np.isnan(r['avg_spearman'])])
    
    print(f'\n=== Final Results ===')
    print(f'Average Validation Loss: {avg_val_loss:.4f}')
    print(f'Average Per-Drug Pearson Correlation: {avg_perdrug_pearson:.4f}')
    print(f'Average Per-Drug Spearman Correlation: {avg_perdrug_spearman:.4f}')

    return models, avg_val_loss

def main_kfold():
    x_gene, x_smiles, y, drug_names, cell_lines, tcga_types = data_prepare(dataDir)
    models, avg_val_loss = train_and_evaluate_kfold(x_gene, x_smiles, y, drug_names, cell_lines, tcga_types, k=5, epochs=epochSize, batch_size=batch_size)

if __name__ == "__main__":
    main_kfold()