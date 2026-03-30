
import sys, os, subprocess

PHASE = os.environ.get('_PHARMA_PHASE', '')

if PHASE == '':
    PYTHON = sys.executable
    SCRIPT = os.path.abspath(__file__)
    env = os.environ.copy()

    print('='*60)
    print('PharmaTFormer LIHC Pipeline')
    print('='*60)

    env['_PHARMA_PHASE'] = 'pretrain'
    print('\nStep 1: Pretraining...')
    r1 = subprocess.run([PYTHON, SCRIPT], env=env)
    if r1.returncode != 0:
        print('Pretrain FAILED'); sys.exit(1)

    env['_PHARMA_PHASE'] = 'finetune'
    print('\nStep 2: Fine-tuning + Evaluation...')
    r2 = subprocess.run([PYTHON, SCRIPT], env=env)
    if r2.returncode != 0:
        print('Finetune FAILED'); sys.exit(1)

    print('\nPipeline complete!')
    sys.exit(0)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random, gc, codecs
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.metrics import roc_curve
from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE
import matplotlib.pyplot as plt

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
random.seed(seed)

dataDir = '/home/zyr/PharmaTFormer/data'
output_dir = '/home/zyr/PharmaTFormer/data'
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = os.path.join(output_dir, 'ensemble_model_LIHC.pth')


class PretrainFeatureExtractor(nn.Module):
    def __init__(self, gene_input_size, gene_hidden_size, drug_hidden_size):
        super().__init__()
        self.gene_fc1 = nn.Linear(gene_input_size, gene_hidden_size)
        self.gene_fc2 = nn.Linear(gene_hidden_size, gene_hidden_size)
        self.smiles_fc = nn.Linear(128, drug_hidden_size)
    def forward(self, gene_expr, smiles):
        g = F.relu(self.gene_fc1(gene_expr))
        g = F.relu(self.gene_fc2(g))
        s = F.relu(self.smiles_fc(smiles))
        return torch.cat((g, s), dim=1)

class PretrainTransModel(nn.Module):
    def __init__(self, feature_dim, nhead, seq_len, dim_feedforward=2048, dropout=0.1, num_layers=3):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output = nn.Sequential(nn.Linear(seq_len * feature_dim, 1024), nn.ReLU(), nn.Dropout(dropout), nn.Linear(1024, 1))
    def forward(self, x):
        return self.output(torch.flatten(self.transformer_encoder(x), 1))

class PretrainCombinedModel(nn.Module):
    def __init__(self, gene_input_size, gene_hidden_size, drug_hidden_size, feature_dim, nhead, num_layers=3, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.feature_extractor = PretrainFeatureExtractor(gene_input_size, gene_hidden_size, drug_hidden_size)
        self.feature_dim = feature_dim
        self.seq_len = (gene_hidden_size + drug_hidden_size) // feature_dim
        self.transformer = PretrainTransModel(feature_dim, nhead, self.seq_len, dim_feedforward, dropout, num_layers)
    def forward(self, gene_expr, smiles):
        f = self.feature_extractor(gene_expr, smiles)
        return self.transformer(f.view(f.size(0), self.seq_len, self.feature_dim))


class FinetuneFeatureExtractor(nn.Module):
    def __init__(self, gene_input_size, gene_hidden_size, drug_hidden_size, dropout=0.2):
        super().__init__()
        self.gene_fc1 = nn.Linear(gene_input_size, gene_hidden_size)
        self.gene_fc2 = nn.Linear(gene_hidden_size, gene_hidden_size)
        self.smiles_fc = nn.Linear(128, drug_hidden_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, gene_expr, smiles):
        g = self.dropout(F.relu(self.gene_fc1(gene_expr)))
        g = self.dropout(F.relu(self.gene_fc2(g)))
        s = self.dropout(F.relu(self.smiles_fc(smiles)))
        return torch.cat((g, s), dim=1)

class FinetuneTransModel(nn.Module):
    def __init__(self, feature_dim, nhead, seq_len, dim_feedforward=2048, dropout=0.2, num_layers=3):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output = nn.Sequential(nn.Linear(seq_len * feature_dim, 1024), nn.ReLU(), nn.Dropout(dropout), nn.Linear(1024, 1))
    def forward(self, x):
        return self.output(torch.flatten(self.transformer_encoder(x), 1))

class FinetuneCombinedModel(nn.Module):
    def __init__(self, gene_input_size, gene_hidden_size, drug_hidden_size, feature_dim, nhead, num_layers=3, dim_feedforward=2048, dropout=0.2):
        super().__init__()
        self.feature_extractor = FinetuneFeatureExtractor(gene_input_size, gene_hidden_size, drug_hidden_size, dropout)
        self.feature_dim = feature_dim
        self.seq_len = (gene_hidden_size + drug_hidden_size) // feature_dim
        self.transformer = FinetuneTransModel(feature_dim, nhead, self.seq_len, dim_feedforward, dropout, num_layers)
    def forward(self, gene_expr, smiles):
        f = self.feature_extractor(gene_expr, smiles)
        return self.transformer(f.view(f.size(0), self.seq_len, self.feature_dim))


def run_pretrain():
    print('='*60)
    print('Phase 1: Pretraining on CCLE ')
    print('='*60)
    batch_size = 128

    def make_loader(x_gene, x_drug, y):
        def worker_init_fn(worker_id):
            np.random.seed(seed); random.seed(seed)
        ds = torch.utils.data.TensorDataset(torch.tensor(x_gene.values).float(), torch.tensor(x_drug).float(), torch.tensor(y.values).float())
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn)

    expr = pd.read_csv(f'{dataDir}/ccle/ccle_rnaseq_LIHC.csv', index_col=0)
    expr.fillna(0, inplace=True)
    expr = expr.apply(lambda x: np.log2(x + 1))
    rsp = pd.read_csv(f'{dataDir}/ccle/PanCancer_drug_response.csv')
    rsp.dropna(inplace=True)
    rsp = rsp[['Cell Line Name', 'Drug Name', 'AUC']].copy()
    smiles_df = pd.read_csv(f'{dataDir}/ccle/drug_smiles.csv')
    smiles_df.set_index('Drug Name', inplace=True)
    smiles_df.index = smiles_df.index.str.strip()
    rsp = rsp[rsp['Drug Name'].isin(smiles_df.index)]
    combined_df = pd.merge(rsp, expr.T, left_on='Cell Line Name', right_index=True)
    combined_df = combined_df.drop_duplicates(subset=['Cell Line Name', 'Drug Name'])
    y = combined_df[['AUC']]
    x_gene = combined_df.drop(columns=['AUC', 'Drug Name', 'Cell Line Name']).apply(pd.to_numeric, errors='coerce').fillna(0)
    bpe_path = f'{dataDir}/bpe.codes'
    with codecs.open(f'{dataDir}/ccle/drug_smiles.csv', encoding='utf-8') as fi, codecs.open(bpe_path, 'w', encoding='utf-8') as fo:
        learn_bpe(fi, fo, num_symbols=10000)
    with codecs.open(bpe_path, encoding='utf-8') as fi:
        bpe = BPE(fi)
    smiles_encoded = []
    for _, row in combined_df.iterrows():
        enc = [ord(c) for c in bpe.process_line(smiles_df.loc[row['Drug Name'], 'SMILES'])]
        if len(enc) > 128: enc = enc[:128]
        smiles_encoded.append(np.pad(enc, (0, 128-len(enc)), 'constant'))
    smiles_encoded = np.array(smiles_encoded)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    models = []
    for fold, (tr, va) in enumerate(kf.split(x_gene)):
        print(f'Fold {fold+1}/5')
        # Per-fold fit: fit scaler on train only, transform both
        x_tr, x_va = x_gene.iloc[tr].copy(), x_gene.iloc[va].copy()
        ss = StandardScaler()
        x_tr = pd.DataFrame(ss.fit_transform(x_tr), columns=x_tr.columns, index=x_tr.index)
        x_va = pd.DataFrame(ss.transform(x_va), columns=x_va.columns, index=x_va.index)
        mm = MinMaxScaler()
        x_tr = pd.DataFrame(mm.fit_transform(x_tr), columns=x_tr.columns, index=x_tr.index)
        x_va = pd.DataFrame(mm.transform(x_va), columns=x_va.columns, index=x_va.index)
        model = PretrainCombinedModel(x_tr.shape[1], 8192, 256, 128, 8).to(device)
        torch.cuda.empty_cache()
        opt = optim.Adam(model.parameters(), lr=0.00001)
        crit = nn.MSELoss()
        train_loader = make_loader(x_tr, smiles_encoded[tr], y.iloc[tr])
        # val_loader created here to maintain random state consistency with original script
        val_loader = make_loader(x_va, smiles_encoded[va], y.iloc[va])
        for ep in range(10):
            model.train(); ls = 0
            for g, s, t in train_loader:
                g, s, t = g.to(device), s.to(device), t.to(device)
                opt.zero_grad(); loss = crit(model(g, s), t); loss.backward(); opt.step(); ls += loss.item()
            print(f'  Epoch {ep+1}/10 - Train Loss: {ls/len(train_loader):.4f}')
        model.eval(); vl = 0
        with torch.no_grad():
            for g, s, t in val_loader:
                g, s, t = g.to(device), s.to(device), t.to(device)
                vl += crit(model(g, s), t).item()
        print(f'  Val Loss: {vl/len(val_loader):.4f}')
        models.append(model)
    torch.save({f'model_{i+1}': m.state_dict() for i, m in enumerate(models)}, MODEL_PATH)
    print(f'Ensemble model saved to {MODEL_PATH}')


def run_finetune():
    ft_batch = 4

    def make_loader(x_gene, x_drug, y):
        def worker_init_fn(worker_id):
            np.random.seed(seed); random.seed(seed)
        ds = torch.utils.data.TensorDataset(torch.tensor(x_gene.loc[y.index].values).float(), torch.tensor(x_drug).float(), torch.tensor(y['AUC'].values).float().unsqueeze(1))
        return torch.utils.data.DataLoader(ds, batch_size=ft_batch, shuffle=True, worker_init_fn=worker_init_fn)

    def load_organoid():
        xg = pd.read_csv(f'{dataDir}/organoid/LIHC/LIHC_organoid_TPM.csv', index_col=0); xg.dropna(inplace=True)
        xg = xg.apply(lambda x: np.log2(x+1)).T
        xg[xg.columns] = StandardScaler().fit_transform(xg)
        xg[xg.columns] = MinMaxScaler().fit_transform(xg)
        y = pd.read_csv(f'{dataDir}/organoid/LIHC/LIHC_drug_response.csv')
        common = xg.index.intersection(y['Organoid Name'])
        xg = xg.loc[common]; y = y[y['Organoid Name'].isin(common)].set_index('Organoid Name').loc[xg.index]
        sdf = pd.read_csv(f'{dataDir}/ccle/drug_smiles.csv'); sdf.set_index('Drug Name', inplace=True); sdf.index = sdf.index.str.strip()
        bpe_f = codecs.open(f'{dataDir}/bpe.codes', encoding='utf-8'); bpe = BPE(bpe_f); bpe_f.close()
        se = []
        for d in y['Drug Name'].values:
            enc = [ord(c) for c in bpe.process_line(sdf.loc[d, 'SMILES'])]
            if len(enc) > 128: enc = enc[:128]
            se.append(np.pad(enc, (0, 128-len(enc)), 'constant'))
        return xg, np.array(se), y

    def load_ensemble(path, gdim):
        st = torch.load(path, weights_only=False); ms = []
        for i in range(len(st)):
            m = FinetuneCombinedModel(gdim, 8192, 256, 128, 8).to(device)
            m.load_state_dict(st[f'model_{i+1}']); ms.append(m)
        return ms

    def finetune(models, xg, xs, y, epochs):
        crit = nn.MSELoss()
        for i, m in enumerate(models):
            print(f'Fine-tuning model {i+1}/{len(models)}')
            m.to(device); torch.cuda.empty_cache()
            opt = optim.Adam(m.parameters(), lr=0.00001, weight_decay=1e-5)
            loader = make_loader(xg, xs, y)
            best_l, best_s, noi = float('inf'), None, 0
            for ep in range(epochs):
                m.train(); ls = 0
                for g, s, t in loader:
                    g, s, t = g.to(device), s.to(device), t.to(device)
                    opt.zero_grad(); loss = crit(m(g, s), t); loss.backward(); opt.step(); ls += loss.item()
                avg = ls/len(loader)
                print(f'  Epoch {ep+1}/{epochs} - Fine-tuning Train Loss: {avg:.4f}')
                if avg < best_l: best_l, best_s, noi = avg, m.state_dict(), 0
                else: noi += 1
                if noi >= 3: print(f'  Early stopping at epoch {ep+1}'); break
            m.load_state_dict(best_s)
        return models

    def evaluate(models, clin_path):
        expr = pd.read_csv(f'{dataDir}/tcga_reproduce/TCGA-LIHC-gene-adjust.csv', index_col=0)
        expr.fillna(0, inplace=True)
        expr = expr.apply(lambda x: np.log2(x+1)).T.apply(pd.to_numeric, errors='coerce').fillna(0)
        expr = pd.DataFrame(StandardScaler().fit_transform(expr), columns=expr.columns, index=expr.index)
        expr = pd.DataFrame(MinMaxScaler().fit_transform(expr), columns=expr.columns, index=expr.index)
        sdf = pd.read_csv(f'{dataDir}/ccle/drug_smiles.csv'); sdf.set_index('Drug Name', inplace=True); sdf.index = sdf.index.str.strip()
        bpe_f = codecs.open(f'{dataDir}/bpe.codes', encoding='utf-8'); bpe = BPE(bpe_f); bpe_f.close()
        enc = [ord(c) for c in bpe.process_line(sdf.loc['Sorafenib', 'SMILES'])]
        if len(enc) > 128: enc = enc[:128]
        se = np.tile(np.pad(enc, (0, 128-len(enc)), 'constant'), (expr.shape[0], 1))
        preds = []
        for m in models:
            m.eval(); fp = []
            dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(expr.values).float(), torch.tensor(se).float()), batch_size=4, shuffle=False)
            with torch.no_grad():
                for g, s in dl: fp.extend(m(g.to(device), s.to(device)).cpu().numpy().flatten())
            preds.append(np.array(fp))
        predictions = np.mean(preds, axis=0)
        rdf = pd.DataFrame({'sample': expr.index, 'Sorafenib_prediction': predictions})
        clin = pd.read_csv(clin_path); clin = clin[['sample','OS','OS.time']].dropna()
        merged = rdf.merge(clin, on='sample')
        fpr, tpr, th = roc_curve(merged['OS'], merged['Sorafenib_prediction'])
        merged['risk_group'] = (merged['Sorafenib_prediction'] > th[np.argmax(tpr-fpr)]).astype(int)
        cph = CoxPHFitter()
        cph.fit(merged[['OS.time','OS','risk_group']], duration_col='OS.time', event_col='OS')
        hr = np.exp(cph.summary['coef'].values[0])
        lo = np.exp(cph.summary['coef lower 95%'].values[0])
        hi = np.exp(cph.summary['coef upper 95%'].values[0])
        p = cph.summary['p'].values[0]
        print(f'Drug: Sorafenib')
        print(f'Hazard Ratio: {hr:.4f} (95% CI: {lo:.4f}-{hi:.4f})')
        print(f'P-value: {p:.6f}')
        kmf = KaplanMeierFitter(); fig, ax = plt.subplots(figsize=(8,6))
        for lab, grp in merged.groupby('risk_group'):
            kmf.fit(grp['OS.time'], grp['OS'], label=f'Risk Group {lab}'); kmf.plot_survival_function(ax=ax)
        plt.title(f'LIHC Sorafenib Overall Survival'); plt.xlabel('Time (days)'); plt.ylabel('Survival Probability')
        plt.savefig(os.path.join(output_dir, f'LIHC_Sorafenib_survival_curve.png')); plt.close()
        merged.to_csv(os.path.join(output_dir, f'LIHC_Sorafenib_predictions_finetune_ensemble.csv'), index=False)

    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    random.seed(seed)

    print('='*60)
    print('Phase 2: Fine-tuning + Evaluation (LIHC Sorafenib)')
    print('='*60)
    xg, xs, y = load_organoid()
    epochs = 15 if xg.shape[0] > 50 else 20
    print(f'Number of samples: {xg.shape[0]}, epochs: {epochs}')
    models = load_ensemble(MODEL_PATH, xg.shape[1])
    models = finetune(models, xg, xs, y, epochs)

    evaluate(models, f'{dataDir}/tcga_reproduce/TCGA_LIHC_merged_Sorafenib_survival_1.csv')

    del models, xg, xs, y; gc.collect(); torch.cuda.empty_cache()
    print('Resources released.')


if PHASE == 'pretrain':
    run_pretrain()
elif PHASE == 'finetune':
    run_finetune()
