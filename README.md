---
title: MetaboNiche
emoji: 🧬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.42.0
app_file: app.py
pinned: false
license: mit
---

# MetaboNiche: Spatial Metabolic Niche Discovery

## Research Objective

MetaboNiche builds and validates a multimodal Graph-Transformer model that integrates **histology (H&E) images**, **spatial coordinates**, and **spot-level gene expression** to identify reproducible, spatially coherent metabolic niches in tissue.

## Features

- **Multi-Sample Processing**: Analyze multiple tissue samples together
- **54 Metabolic Pathways**: Comprehensive metabolic profiling
- **Graph Transformer**: Spatial-aware niche discovery
- **SHAP Analysis**: Gene importance interpretation
- **AI Explanations**: GPT-4 powered insights

## Data Format

Upload `.tar.gz` or `.zip` archives containing 10X Visium data:
```
Count Matrix Archive:
├── filtered_feature_bc_matrix/
│   ├── matrix.mtx.gz
│   ├── features.tsv.gz
│   └── barcodes.tsv.gz

Spatial Archive:
├── spatial/
│   ├── tissue_positions_list.csv
│   └── tissue_hires_image.png
```

## Default Samples

- 1142243F, 1160920F, CID4290, CID4465, CID4535, CID44971

## Data Source

The data used in this project is available at:
- [Google Drive - MetaboNiche Data](https://drive.google.com/drive/folders/1C9m4YvR4laTT4z0g9VNjLUq2nXhXBgEx?usp=sharing)

Model pretraining was performed using:
- [Hest-1k Dataset](https://huggingface.co/datasets/MahmoodLab/hest) - Spatial transcriptomics dataset from HuggingFace

## Usage

1. **Load Default**: Process all pre-configured samples
2. **Load Pre-computed**: View saved results instantly
3. **Upload Custom**: Analyze your own data
