# PharmaFormer: Organoid-Guided Transfer Learning for Predicting Clinical Drug Response

**Authors**: Yuru Zhou, Quanhui Dai, Yanming Xu, Shuang Wu, Minzhang Cheng,* and Bing Zhao,*

## Overview

PharmaFormer is a **Transformer-based deep learning model** designed to predict clinical drug responses by integrating gene expression profiles and drug molecular structures. This model combines patient-derived tumor organoid data with large-scale cell line pharmacogenomic datasets, leveraging a transfer learning approach to improve prediction accuracy.

<p align="center">
  <img src="Schematic of the PharmaFormer model.jpg" alt="PharmaFormer Architecture" width="500"/>
</p>

## Key Features

1. **Seamless Transfer Learning**: PharmaFormer leverages a pre-trained model developed on extensive cell line data (87,596 cell line-drug pairs) from the GDSC2 dataset and fine-tunes it with organoid-specific data, enhancing its ability to predict patient drug responses with high accuracy.
2. **Leveraging Organoid Insights**: By utilizing the biomimetic fidelity of organoids, which closely mirror clinical drug responses, PharmaFormer highlights the critical role of organoid data in advancing drug response prediction models.
3. **Enhanced Predictive Performance**: PharmaFormer excels in predicting drug responses, particularly for clinical tumor patients, providing robust and accurate predictions that support personalized oncology and precision medicine efforts.

## Installation

```bash
git clone https://github.com/yourusername/PharmaFormer.git
cd PharmaFormer
pip install -r requirements.txt

