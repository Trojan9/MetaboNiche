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

## Deployed Application

The application is live and accessible at: [https://huggingface.co/spaces/Trojan9/metaboniche](https://huggingface.co/spaces/Trojan9/metaboniche)

## Folder Structure

```
metaboniche/
├── app.py                          # Main Gradio application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── assets/                         # Application assets
│   └── metabolic_genes_vocab.json  # Metabolic gene vocabulary
│
├── models/                         # Trained model weights
│   ├── best_graph.pt              # Graph Transformer model
│   └── best_finetune.pt           # Fine-tuned multimodal model
│
├── default_files/                  # Pre-loaded tissue samples (6 samples)
│   ├── {SAMPLE}_filtered_count_matrix.tar.gz  # Gene expression data
│   ├── {SAMPLE}_spatial.tar.gz                # Spatial coordinates & H&E images
│   └── {SAMPLE}_metadata.tar.gz               # Sample metadata
│
├── tissue_images/                  # Tissue H&E images for visualization
│   ├── 1142243F.png
│   ├── 1160920F.png
│   ├── CID4290.png
│   ├── CID4465.png
│   ├── CID44971.png
│   └── CID4535.png
│
├── images/                         # Sample interfaces and pipeline diagrams
│   ├── pipeline.png                # Model architecture/pipeline visualization
│   ├── hf_prediction_output.png    # Example prediction output
│   └── Screenshot 2026-01-06 at 11.50.37.png  # Interface screenshot
│
├── results/                        # Pre-computed analysis results
│   ├── graph_predictions.csv       # Graph model predictions
│   ├── graph_shap.csv             # SHAP values for graph model
│   ├── graph_spatial_data.csv     # Spatial data with predictions
│   ├── graph_minmax.csv           # Normalization parameters
│   ├── mm_metabolic_predictions.csv  # Multimodal predictions
│   ├── mm_shap.csv                # SHAP values for multimodal model
│   ├── mm_spatial_data.csv        # Multimodal spatial data
│   └── mm_minmax.csv              # Multimodal normalization parameters
│
└── training/                       # Training notebooks and experiments
    ├── training_and_deployment.ipynb           # Model training pipeline
    └── Comparison_between_graph_and_mm.ipynb   # Model comparison analysis
```
