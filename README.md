PharmaFormer: Organoid-Guided Transfer Learning for Predicting Clinical Drug Response
Authors: Yuru Zhou, Minzhang Cheng, Quanhui Dai, Bing Zhao
Affiliations:

School of Basic Medical Sciences, The First Affiliated Hospital of Nanchang University, Nanchang, China
Z Lab, bioGenous BIOTECH, Shanghai, China
Overview
PharmaFormer is a Transformer-based deep learning model designed to predict clinical drug responses by integrating gene expression profiles and drug molecular structures. This model combines patient-derived tumor organoid data with large-scale cell line pharmacogenomic datasets, leveraging a transfer learning approach to improve prediction accuracy. The model offers a robust and scalable tool for advancing personalized oncology, with potential applications in both drug development and clinical decision-making.

Key Features
Transfer Learning from Cell Lines to Organoids: PharmaFormer uses a pre-trained model built on extensive cell line data (87,596 cell line-drug pairs) from the GDSC2 dataset, fine-tuned with organoid-specific data to achieve high predictive fidelity for patient drug responses.

Transformer Architecture: Utilizing a custom Transformer encoder with self-attention and position-wise feedforward networks, PharmaFormer effectively captures complex interactions between gene expression data and drug structure, making it suitable for predicting drug responses in diverse tumor types.

Enhanced Clinical Relevance: PharmaFormer’s predictions allow for patient stratification into drug-sensitive and drug-resistant groups, verified by survival analysis and Cox proportional hazard ratios. This model has demonstrated superior performance in predicting responses to key drugs across colon, bladder, and liver cancer cohorts.

Installation
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/PharmaFormer.git
cd PharmaFormer
Install required dependencies:

bash
Copy code
pip install -r requirements.txt
Required libraries include:

torch for PyTorch-based model development
cuml for GPU-accelerated machine learning benchmarks
scikit-learn and lifelines for preprocessing and survival analysis
Usage
Data Preparation
Cell Line Data: Obtain gene expression and drug response data from Cell Model Passport and GDSC2.
Organoid Data: Collect organoid drug response and gene expression data from previously published studies.
Clinical Data: Download tumor tissue gene expression, survival data, and drug treatment information from TCGA.
Model Training
PharmaFormer training involves two main stages:

Pre-training: Use GDSC2 cell line data for initial training.
Fine-tuning: Apply organoid-specific datasets for transfer learning. Fine-tuning parameters are configured in the config.yaml file.
To start training:

bash
Copy code
python train.py --config config.yaml
Model Evaluation
Evaluate PharmaFormer’s performance with both pre-trained and fine-tuned models:

bash
Copy code
python evaluate.py --config config.yaml
Performance metrics include:

Pearson and Spearman correlations for drug response predictions
Kaplan-Meier survival analysis for clinical relevance
Cox proportional hazards for risk stratification
Benchmarking Against Classical Models
For comparison, PharmaFormer is evaluated alongside classical machine learning models:

Support Vector Regression (SVR)
Multi-Layer Perceptron (MLP)
Random Forest (RF)
k-Nearest Neighbors (KNN)
Ridge Regression
These models are implemented with cuml for GPU acceleration and can be trained with the following:

bash
Copy code
python benchmark.py --config config.yaml
Results
PharmaFormer demonstrated enhanced predictive accuracy over classical models, especially in clinical drug response predictions for colon, bladder, and liver cancer patients. Survival analysis confirmed PharmaFormer’s ability to stratify patients into distinct risk categories based on predicted drug sensitivity.

Citation
If you use PharmaFormer in your research, please cite:

scss
Copy code
@article{Zhou2024PharmaFormer,
  title={Organoid-Guided Transfer Learning for Predicting Clinical Drug Response},
  author={Zhou, Yuru and Cheng, Minzhang and Dai, Quanhui and Zhao, Bing},
  year={2024}
}
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
This work was supported by grants from the National Natural Science Foundation of China, the Key Research and Development Program of Jiangxi Province, and the Natural Science Foundation of Shandong Province.
