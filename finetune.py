import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import joblib
import random
import os
import gc
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.metrics import roc_curve
from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE
import codecs
import matplotlib.pyplot as plt 

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
random.seed(seed)

# General configuration
dataDir = '/home/zyr/PharmaTFormer/data'
batch_size = 4
lr = 0.00001
output_dir = "/home/zyr/PharmaTFormer/data"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class FeatureExtractor(nn.Module):
    def __init__(self, gene_input_size, gene_hidden_size, drug_hidden_size, dropout=0.3 ):
        super(FeatureExtractor, self).__init__()
        self.gene_fc1 = nn.Linear(gene_input_size, gene_hidden_size)
        self.gene_fc2 = nn.Linear(gene_hidden_size, gene_hidden_size)
        self.smiles_fc = nn.Linear(128, drug_hidden_size)
        self.dropout = nn.Dropout(dropout)  # Adding Dropout

    def forward(self, gene_expr, smiles):
        gene_out = F.relu(self.gene_fc1(gene_expr))
        gene_out = self.dropout(gene_out)  # Apply Dropout
        gene_out = F.relu(self.gene_fc2(gene_out))
        gene_out = self.dropout(gene_out)  # Apply Dropout
        smiles_out = F.relu(self.smiles_fc(smiles))
        smiles_out = self.dropout(smiles_out)  # Apply Dropout
        combined_features = torch.cat((gene_out, smiles_out), dim=1)
        return combined_features

class TransModel(nn.Module):
    def __init__(self, feature_dim, nhead, seq_len, dim_feedforward=2048, dropout=0.3, num_layers=3):
        super(TransModel, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Sequential(
            nn.Linear(seq_len * feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),  # Adding Dropout
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = torch.flatten(x, 1)
        return self.output(x)

class CombinedModel(nn.Module):
    def __init__(self, gene_input_size, gene_hidden_size, drug_hidden_size, feature_dim, nhead, num_layers=3, dim_feedforward=2048, dropout=0.3):
        super(CombinedModel, self).__init__()
        self.feature_extractor = FeatureExtractor(gene_input_size, gene_hidden_size, drug_hidden_size, dropout=dropout)
        self.feature_dim = feature_dim
        self.seq_len = (gene_hidden_size + drug_hidden_size) // feature_dim  # Calculate appropriate seq_len
        self.transformer = TransModel(feature_dim=feature_dim, nhead=nhead, seq_len=self.seq_len, num_layers=num_layers, dim_feedforward=dim_feedforward, dropout=dropout)

    def forward(self, gene_expr, smiles):
        features = self.feature_extractor(gene_expr, smiles)

        # Ensure features can be reshaped to (batch_size, seq_len, feature_dim)
        batch_size = features.size(0)
        features = features.view(batch_size, self.seq_len, self.feature_dim)

        output = self.transformer(features)
        return output

def prepare_data(x_gene, x_drug, y, batch_size=batch_size, seed=seed):
    def worker_init_fn(worker_id):
        np.random.seed(seed)
        random.seed(seed)

    repeated_x_gene_tensor = torch.tensor(x_gene.loc[y.index].values).float()
    
    x_drug_tensor = torch.tensor(x_drug).float()
    y_tensor = torch.tensor(y['AUC'].values).float().unsqueeze(1)

    # Ensure all tensors align on the first dimension
    if not (repeated_x_gene_tensor.size(0) == x_drug_tensor.size(0) == y_tensor.size(0)):
        raise ValueError("Tensors have mismatched sizes!")

    dataset = torch.utils.data.TensorDataset(repeated_x_gene_tensor, x_drug_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn)
    return dataloader

def load_organoid_data(gene_matrix_path, drug_response_path, cancer_type):
    # Load gene matrix data
    x_gene = pd.read_csv(gene_matrix_path, index_col=0)
    x_gene.dropna(inplace=True)
    x_gene = x_gene.apply(lambda x: np.log2(x + 1))

    # Transpose the data so that rows represent samples and columns represent genes
    x_gene = x_gene.T

    # Standardize and normalize gene expression data
    gene_columns = x_gene.columns
    standard_scaler = StandardScaler()
    x_gene[gene_columns] = standard_scaler.fit_transform(x_gene[gene_columns])

    min_max_scaler = MinMaxScaler()
    x_gene[gene_columns] = min_max_scaler.fit_transform(x_gene[gene_columns])

    # Load drug response data
    y = pd.read_csv(drug_response_path)

    # Check necessary columns
    if 'Drug Name' not in y.columns or 'Organoid Name' not in y.columns:
        raise KeyError("The required columns 'Drug Name' and 'Organoid Name' are missing in the drug response data.")

    # Align data by matching sample names
    common_samples = x_gene.index.intersection(y['Organoid Name'])
    if len(common_samples) != len(x_gene.index):
        print(f"Warning: {len(x_gene.index) - len(common_samples)} samples in x_gene do not have matching entries in y.")

    # Filter x_gene and y based on common samples
    x_gene = x_gene.loc[common_samples]
    y = y[y['Organoid Name'].isin(common_samples)].set_index('Organoid Name').loc[x_gene.index]

    # Load SMILES data and process SMILES encoding
    smiles_df = pd.read_csv(f'{dataDir}/ccle/drug_smiles.csv')
    smiles_df.set_index('Drug Name', inplace=True)
    smiles_df.index = smiles_df.index.str.strip()

    bpe_codes_path = f'{dataDir}/bpe.codes'
    with codecs.open(bpe_codes_path, encoding='utf-8') as f_in:
        bpe = BPE(f_in)

    smiles_encoded = []
    for drug in y['Drug Name'].values:
        if drug not in smiles_df.index:
            raise KeyError(f"Drug '{drug}' not found in SMILES data.")
        encoded_smile = [ord(char) for char in bpe.process_line(smiles_df.loc[drug, 'SMILES'])]
        if len(encoded_smile) > 128:
            encoded_smile = encoded_smile[:128]
        padded_smile = np.pad(encoded_smile, (0, 128 - len(encoded_smile)), 'constant')
        smiles_encoded.append(padded_smile)

    smiles_encoded = np.array(smiles_encoded)

    return x_gene, smiles_encoded, y

def fine_tune_ensemble_model(models, x_gene, x_smiles, y, cancer_type, drug, epochs, lr=lr, batch_size=batch_size, patience=3):
    criterion = nn.MSELoss()

    for i, model in enumerate(models):
        print(f"Fine-tuning model {i + 1}/{len(models)}")
        model = model.to(device)
        torch.cuda.empty_cache()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3) 

        train_loader = prepare_data(x_gene, x_smiles, y, batch_size, seed=seed)

        best_loss = float('inf')
        best_model_state = None
        epochs_no_improve = 0

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
            print(f'Epoch {epoch + 1}/{epochs} - Fine-tuning Train Loss: {train_loss:.4f}')

            # Early Stopping mechanism
            if train_loss < best_loss:
                best_loss = train_loss
                best_model_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        # Restore to best model state
        model.load_state_dict(best_model_state)

        model_save_path = os.path.join(output_dir, f"fine_tuned_model_{cancer_type}_{drug}_ensemble_{i + 1}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Fine-tuned model {i + 1} for {cancer_type} and {drug} saved to {model_save_path}")

    return models

def load_ensemble_model(save_path, gene_input_size, gene_hidden_size, drug_hidden_size, feature_dim, nhead):
    ensemble_state_dict = torch.load(save_path)
    models = []

    # Assume that the saved state dictionary is a collection of models
    for i in range(len(ensemble_state_dict)):
        model = CombinedModel(gene_input_size=gene_input_size, gene_hidden_size=gene_hidden_size, drug_hidden_size=drug_hidden_size, feature_dim=feature_dim, nhead=nhead)
        model.load_state_dict(ensemble_state_dict[f'model_{i + 1}'])  
        model = model.to(device)
        models.append(model)
    
    return models

def predict_with_ensemble(models, gene_expr, smiles, batch_size=4):
    predictions = []

    for model in models:
        model.eval()
        gene_expr = gene_expr.apply(pd.to_numeric, errors='coerce').fillna(0)
        data_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(gene_expr.values).float(),
                torch.tensor(smiles).float()
            ),
            batch_size=batch_size, shuffle=False
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

def evaluate_on_clinical_data(models, cancer_type, drug, tcga_data_path):
    tcga_expr_processed, smiles_encoded = process_tcga_data(tcga_data_path, cancer_type=cancer_type, drug_name=drug)
    
    predictions = predict_with_ensemble(models, tcga_expr_processed, smiles_encoded)

    result_df = pd.DataFrame({
        'sample': tcga_expr_processed.index,
        f'{drug}_prediction': predictions
    })

    clinical_data = load_clinical_data(drug, cancer_type=cancer_type)

    merged_data = result_df.merge(clinical_data, on='sample')

    fpr, tpr, thresholds = roc_curve(merged_data['OS'], merged_data[f'{drug}_prediction'])
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    merged_data['risk_group'] = (merged_data[f'{drug}_prediction'] > optimal_threshold).astype(int)

    # Perform Cox regression analysis
    cph = CoxPHFitter()
    cph.fit(merged_data[['OS.time', 'OS', 'risk_group']], duration_col='OS.time', event_col='OS')
    hr = np.exp(cph.summary['coef'].values[0])
    hr_ci_lower = np.exp(cph.summary['coef lower 95%'].values[0])
    hr_ci_upper = np.exp(cph.summary['coef upper 95%'].values[0])
    p_value = cph.summary['p'].values[0]

    print(f"Drug: {drug}")
    print(f"Optimal threshold: {optimal_threshold}")
    print(f"Hazard Ratio: {hr} (95% CI: {hr_ci_lower}-{hr_ci_upper})")
    print(f"P-value: {p_value}\n")

    kmf = KaplanMeierFitter()
    T = merged_data['OS.time']
    E = merged_data['OS']
    groups = merged_data['risk_group']
    ix = (groups == 0)

    plt.figure(figsize=(8,6))
    kmf.fit(T[ix], E[ix], label='Low Risk')
    ax = kmf.plot_survival_function()

    kmf.fit(T[~ix], E[~ix], label='High Risk')
    kmf.plot_survival_function(ax=ax)

    plt.title(f'Survival Curves for {cancer_type} patients treated with {drug}')
    plt.xlabel('Time (days)')
    plt.ylabel('Survival Probability')
    plt.legend()
    plt.grid(True)

    plot_save_path = os.path.join(output_dir, f"{cancer_type}_{drug}_survival_curve.png")
    plt.savefig(plot_save_path)
    plt.close()
    print(f"Survival curve saved to {plot_save_path}")

    output_columns = ['sample', f'{drug}_prediction', 'risk_group', 'OS', 'OS.time', 'PFI', 'PFI.time']
    output_data = merged_data[output_columns]
    result_save_path = os.path.join(output_dir, f"{cancer_type}_{drug}_predictions_finetune_ensemble.csv")
    output_data.to_csv(result_save_path, index=False)
    print(f"Predictions with clinical data saved to {result_save_path}")

def process_tcga_data(data_path, cancer_type='BLCA', drug_name='Gemcitabine', dataDir='/home/zyr/PharmaTFormer/data', standardization=True, normalization=True):
    expr = pd.read_csv(data_path, index_col=0, header=0)
    expr.dropna(inplace=True)
    expr = expr.apply(lambda x: np.log2(x + 1))
    expr = expr.T

    gene_columns = [col for col in expr.columns]
    expr[gene_columns] = expr[gene_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

    if standardization:
        scaler = StandardScaler()
        expr = pd.DataFrame(scaler.fit_transform(expr), columns=expr.columns, index=expr.index)

    if normalization:
        scaler = MinMaxScaler()
        expr = pd.DataFrame(scaler.fit_transform(expr), columns=expr.columns, index=expr.index)

    # Get SMILES representation of the drug and encode
    smiles_df = pd.read_csv(f'{dataDir}/ccle/drug_smiles.csv')
    smiles_df.set_index('Drug Name', inplace=True)
    smiles_df.index = smiles_df.index.str.strip()

    if drug_name not in smiles_df.index:
        raise ValueError(f"Drug {drug_name} not found in SMILES data.")

    # Encode SMILES data using BPE
    bpe_codes_path = f'{dataDir}/bpe.codes'
    with codecs.open(bpe_codes_path, encoding='utf-8') as f_in:
        bpe = BPE(f_in)

    encoded_smile = [ord(char) for char in bpe.process_line(smiles_df.loc[drug_name, 'SMILES'])]
    if len(encoded_smile) > 128:
        encoded_smile = encoded_smile[:128]
    smiles_encoded = np.pad(encoded_smile, (0, 128 - len(encoded_smile)), 'constant')
    
    # Duplicate SMILES encoding to match the number of samples
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

def fine_tune_and_evaluate(cancer_type, drug, gene_matrix_path, drug_response_path, tcga_data_path):
    # Reset random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    models = None

    # Load data and fine-tune model
    try:
        x_gene, smiles_encoded, y = load_organoid_data(gene_matrix_path, drug_response_path, cancer_type)

        num_samples = x_gene.shape[0]
        print(f"Number of samples for {cancer_type}: {num_samples}")
        if num_samples > 50:
            epochs = 15
        else:
            epochs = 20
        print(f"Setting epochs to {epochs} for {cancer_type}")

        # Load pre-trained ensemble model
        ensemble_model_path = os.path.join(dataDir, "ensemble_model-1.pth")
        models = load_ensemble_model(ensemble_model_path, gene_input_size=x_gene.shape[1], gene_hidden_size=8192, drug_hidden_size=256, feature_dim=128, nhead=8)

        # Fine-tune the ensemble models
        models = fine_tune_ensemble_model(models, x_gene, smiles_encoded, y, cancer_type, drug, epochs=epochs)

        # Evaluate on TCGA data
        evaluate_on_clinical_data(models, cancer_type, drug, tcga_data_path)
    finally:
        # Release resources
        if models is not None:
            del models
        del x_gene
        del smiles_encoded
        del y
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Resources for cancer type {cancer_type} and drug {drug} have been released.")

def main():
    # Set cancer types and corresponding drugs
    cancer_drug_info = {
        'COAD': {
            'gene_matrix_path': '/home/zyr/PharmaTFormer/data/crc-organoid-tpm-2.csv',
            'drug_response_path': '/home/zyr/PharmaTFormer/data/merge/COAD-drug.csv',
            'tcga_data_path': f'{dataDir}/TCGA-COAD-gene-adjust-2.csv',
            'drugs': ["5-Fluorouracil", "Oxaliplatin"]
        },
        'BLCA': {
            'gene_matrix_path': '/home/zyr/PharmaTFormer/data/bladder-organoid-tpm-2.csv',
            'drug_response_path': '/home/zyr/PharmaTFormer/data/merge/filtered_BLCA-drug-3.csv',
            'tcga_data_path': f'{dataDir}/TCGA-BLCA-gene-adjust-2.csv',
            'drugs': ["Gemcitabine", "Cisplatin"]
        }
    }

    # Fine-tune and evaluate for each drug and cancer type
    for cancer_type, info in cancer_drug_info.items():
        for drug in info['drugs']:
            fine_tune_and_evaluate(
                cancer_type, drug,
                gene_matrix_path=info['gene_matrix_path'],
                drug_response_path=info['drug_response_path'],
                tcga_data_path=info['tcga_data_path']
            )

if __name__ == "__main__":
    main()
