# PharmaFormer: Organoid-Guided Transfer Learning for Predicting Clinical Drug Response

**Authors**: Yuru Zhou, Quanhui Dai, Yanming Xu, Shuang Wu, Minzhang Cheng,* and Bing Zhao,*

## Overview

PharmaFormer is a **Transformer-based deep learning model** designed to predict clinical drug responses by integrating gene expression profiles and drug molecular structures. This model combines patient-derived tumor organoid data with large-scale cell line pharmacogenomic datasets, leveraging a transfer learning approach to improve prediction accuracy.

<p align="center">
  <img src="PharmaFormer_architecture.jpg" alt="PharmaFormer Architecture" width="500"/>
</p>

## Key Features

1. **Transfer Learning from Cell Lines to Organoids**: PharmaFormer uses a pre-trained model built on extensive cell line data (87,596 cell line-drug pairs) from the GDSC2 dataset, fine-tuned with organoid-specific data to achieve high predictive fidelity for patient drug responses.
2. **Importance of Organoid Data**: Organoids are known for accurately reflecting clinical drug responses due to their biomimetic characteristics. Our study underscores the value of incorporating organoid data to enhance prediction models.
3. **Superior Predictive Capabilities**: PharmaFormer demonstrates improved prediction accuracy, especially for clinical tumor drug response prediction.

## Installation

```bash
git clone https://github.com/yourusername/PharmaFormer.git
cd PharmaFormer
pip install -r requirements.txt

