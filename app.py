import os, tarfile, zipfile, tempfile, warnings, json, shutil, traceback, time, sys, re, base64, io
from datetime import datetime
warnings.filterwarnings('ignore')
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from scipy import sparse
from scipy.io import mmread
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.neighbors import NearestNeighbors

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import TransformerConv
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

PROJECT_ROOT = Path(".").resolve()
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"
OUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_FILES_DIR = PROJECT_ROOT / "default_files"
RESULTS_DIR = PROJECT_ROOT / "results"
IMAGES_DIR = PROJECT_ROOT / "tissue_images"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "best_finetune.pt"
VOCAB_PATH = ASSETS_DIR / "metabolic_genes_vocab.json"
GRAPH_PATH = MODELS_DIR / "best_graph.pt"

# ============ FILE DETECTION HELPERS ============

def is_gzipped(filepath):
    """Check if a file is actually gzipped by inspecting magic bytes."""
    try:
        with open(filepath, 'rb') as f:
            magic = f.read(2)
        return magic == b'\x1f\x8b'
    except:
        return False

def read_mtx_file(filepath):
    """Read a Matrix Market (.mtx) file, handling fake .gz files."""
    filepath = Path(filepath)
    if filepath.suffix == ".gz" and not is_gzipped(filepath):
        temp_path = Path(tempfile.gettempdir()) / filepath.stem
        shutil.copy2(filepath, temp_path)
        result = mmread(str(temp_path)).tocsr()
        temp_path.unlink()
        return result
    else:
        return mmread(str(filepath)).tocsr()

def read_tsv_file(filepath, **kwargs):
    """Read a TSV file, handling fake .gz input."""
    filepath = Path(filepath)
    if filepath.suffix == ".gz" and not is_gzipped(filepath):
        temp_path = Path(tempfile.gettempdir()) / filepath.stem
        shutil.copy2(filepath, temp_path)
        result = pd.read_csv(temp_path, sep="\t", header=None, **kwargs)
        temp_path.unlink()
        return result
    else:
        return pd.read_csv(filepath, sep="\t", header=None, **kwargs)

def find_file(directory, possible_names):
    """Find a file from a list of possible names, searching recursively."""
    directory = Path(directory)
    for name in possible_names:
        path = directory / name
        if path.exists():
            return path
        matches = list(directory.rglob(name))
        if matches:
            return matches[0]
    return None

def find_directory(base_dir, possible_patterns):
    """Find a directory matching any of the patterns."""
    base_dir = Path(base_dir)
    for pattern in possible_patterns:
        for item in base_dir.iterdir():
            if item.is_dir():
                if pattern in item.name.lower():
                    return item
        for item in base_dir.rglob("*"):
            if item.is_dir() and pattern in item.name.lower():
                return item
    return None

def extract_archive(archive_path, dest_dir):
    """Extract tar.gz or zip archive."""
    archive_path = Path(archive_path)
    if archive_path.suffix == ".zip" or archive_path.name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, 'r') as zf:
            zf.extractall(dest_dir)
    elif ".tar" in archive_path.name or archive_path.name.endswith(".gz"):
        with tarfile.open(archive_path, "r:*") as tar:
            tar.extractall(dest_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.name}")

# File name patterns for discovery
MATRIX_NAMES = ["matrix.mtx.gz", "matrix.mtx"]
FEATURES_NAMES = ["features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv"]
BARCODES_NAMES = ["barcodes.tsv.gz", "barcodes.tsv"]
POSITIONS_NAMES = ["tissue_positions_list.csv", "tissue_positions.csv", "tissue_positions_list.txt", "tissue_positions.txt"]
HIRES_IMAGE_NAMES = ["tissue_hires_image.png", "tissue_hires.png"]
LOWRES_IMAGE_NAMES = ["tissue_lowres_image.png", "tissue_lowres.png"]
SCALEFACTORS_NAMES = ["scalefactors_json.json", "scalefactors.json"]

class Logger:
    def __init__(self):
        self.logs = []
    def log(self, msg, level="INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] [{level}] {msg}"
        self.logs.append(log_entry)
        print(log_entry)
        sys.stdout.flush()
        return log_entry
    def get_all(self):
        return "\n".join(self.logs)
    def error(self, msg, exc=None):
        self.log(msg, "ERROR")
        if exc:
            tb = traceback.format_exc()
            self.log(f"Exception details:\n{tb}", "ERROR")
        return self.get_all()

PATHWAYS = ['Glycolysis','TCA_Cycle','Oxidative_Phosphorylation','Fatty_Acid_Oxidation','Fatty_Acid_Synthesis','Pentose_Phosphate_Pathway','Amino_Acid_Metabolism','One_Carbon_Metabolism','Glutamine_Metabolism','Serine_Glycine_Metabolism','Branched_Chain_Amino_Acids','Aromatic_Amino_Acids','Arginine_Metabolism','Proline_Metabolism','Cysteine_Metabolism','Threonine_Metabolism','Histidine_Metabolism','Fatty_Acid_Elongation','Fatty_Acid_Desaturation','Carnitine_Shuttle','Cholesterol_Synthesis','Cholesterol_Efflux','Sphingolipid_Metabolism','Ceramide_Metabolism','Purine_Synthesis','Pyrimidine_Synthesis','Nucleotide_Salvage','Nucleotide_Degradation','Ribonucleotide_Reductase','Folate_Cycle','Methionine_Cycle','SAM_Cycle','One_Carbon_Mitochondrial','Glutathione_Metabolism','NADPH_Production','ROS_Defense','Thioredoxin_System','Peroxide_Metabolism','Heme_Synthesis','Phospholipid_Synthesis','Triglyceride_Synthesis','Acetyl_CoA_Metabolism','Polyamine_Synthesis','Ketone_Body_Metabolism','Glycan_Synthesis','Proteasome_Activity','Autophagy','Glucose_Transport','Amino_Acid_Transport','Lactate_Transport','Mitochondrial_Transport','Lactate_Metabolism','Pyruvate_Metabolism','Gluconeogenesis']

COLOR_PALETTES = {
    'RdBu_r': 'Red-Blue (Diverging)',
    'Viridis': 'Viridis (Sequential)',
    'Plasma': 'Plasma (Sequential)',
    'YlOrRd': 'Yellow-Orange-Red',
    'Blues': 'Blues (Sequential)',
    'Greens': 'Greens (Sequential)',
    'RdYlBu_r': 'Red-Yellow-Blue',
    'Spectral_r': 'Spectral (Diverging)'
}

CELL_TYPE_MARKERS = {
    'Cancer_Epithelial': ['EPCAM','KRT8','KRT18','KRT19','MUC1','CDH1'],
    'Luminal_Epithelial': ['KRT8','KRT18','GATA3','FOXA1','ESR1','PGR'],
    'Basal_Epithelial': ['KRT5','KRT14','TP63','ITGA6','ACTA2'],
    'Myoepithelial': ['ACTA2','KRT14','KRT5','MYH11','OXTR'],
    'HER2_Enriched': ['ERBB2','GRB7'],
    'Fibroblasts': ['COL1A1','COL1A2','VIM','FAP','DCN','LUM'],
    'CAFs': ['FAP','PDPN','PDGFRA','ACTA2','COL1A1'],
    'Myofibroblasts': ['ACTA2','TAGLN','MYL9','TPM2'],
    'T_Cells': ['CD3D','CD3E','CD3G','CD2','TRAC'],
    'CD4_T_Cells': ['CD3D','CD4','IL7R'],
    'CD8_T_Cells': ['CD3D','CD8A','CD8B'],
    'Regulatory_T_Cells': ['FOXP3','IL2RA','CTLA4','CD4'],
    'Exhausted_T_Cells': ['PDCD1','HAVCR2','LAG3','TIGIT'],
    'Cytotoxic_T_Cells': ['GZMA','GZMB','PRF1','NKG7','CD8A'],
    'Memory_T_Cells': ['IL7R','CCR7','SELL','TCF7'],
    'Naive_T_Cells': ['CCR7','SELL','TCF7','LEF1'],
    'B_Cells': ['CD79A','CD79B','MS4A1','CD19','BANK1'],
    'Plasma_Cells': ['MZB1','SDC1','CD38','JCHAIN','IGHG1'],
    'Naive_B_Cells': ['IGHD','TCL1A','FCER2','MS4A1'],
    'Memory_B_Cells': ['MS4A1','CD27','TNFRSF13B'],
    'Myeloid_Cells': ['LYZ','CD68','CD14','FCGR3A'],
    'Macrophages': ['CD68','CD163','MSR1','MRC1','C1QA','C1QB'],
    'M1_Macrophages': ['CD68','CD86','IL1B','TNF','NOS2'],
    'M2_Macrophages': ['CD68','CD163','MRC1','MSR1','IL10'],
    'TAMs': ['CD68','CD163','C1QA','APOE','SPP1'],
    'Monocytes': ['CD14','FCGR3A','S100A8','S100A9','LYZ'],
    'Classical_Monocytes': ['CD14','S100A8','S100A9','LYZ'],
    'Non_Classical_Monocytes': ['FCGR3A','MS4A7','CDKN1C'],
    'Dendritic_Cells': ['CD1C','FCER1A','CLEC10A'],
    'Myeloid_DCs': ['CD1C','FCER1A','CLEC10A'],
    'Plasmacytoid_DCs': ['IL3RA','CLEC4C','LILRA4','IRF7'],
    'NK_Cells': ['NCAM1','NKG7','GNLY','KLRD1','KLRF1'],
    'Mast_Cells': ['TPSAB1','TPSB2','CPA3','KIT'],
    'Neutrophils': ['S100A8','S100A9','CSF3R','FCGR3B'],
    'Endothelial': ['PECAM1','VWF','CDH5','ENG','CLDN5'],
    'Vascular_Endothelial': ['PECAM1','VWF','CDH5','ENG'],
    'Lymphatic_Endothelial': ['LYVE1','PROX1','FLT4','PDPN'],
    'Tip_Endothelial': ['ESM1','ANGPT2','APLN'],
    'Pericytes': ['RGS5','ACTA2','PDGFRB','CSPG4','NOTCH3'],
    'Vascular_Smooth_Muscle': ['ACTA2','MYH11','TAGLN','CNN1'],
    'Adipocytes': ['ADIPOQ','LEP','PLIN1','FABP4','LPL'],
    'Cycling_Cells': ['MKI67','TOP2A','PCNA','CDK1'],
    'Epithelial_General': ['EPCAM','KRT8','KRT18','CDH1'],
    'Stromal_Cells': ['VIM','DCN','LUM','COL1A1'],
}

PATHWAY_GENES = {
    'Glycolysis': ['HK1','HK2','GCK','PFKL','PFKM','PFKP','ALDOA','ALDOB','ALDOC','TPI1','GAPDH','PGK1','PGK2','PGAM1','PGAM2','ENO1','ENO2','ENO3','PKM','PKLR','LDHA','LDHB'],
    'TCA_Cycle': ['CS','ACO1','ACO2','IDH1','IDH2','IDH3A','OGDH','SUCLA2','SUCLG1','SUCLG2','SDHA','SDHB','SDHC','SDHD','FH','MDH1','MDH2'],
    'Oxidative_Phosphorylation': ['NDUFA1','NDUFA2','NDUFB1','NDUFB2','NDUFS1','NDUFS2','SDHA','SDHB','UQCR','COX4I1','COX5A','COX5B','ATP5A1','ATP5B','ATP5C1','ATP5F1A','ATP5F1B'],
    'Fatty_Acid_Oxidation': ['CPT1A','CPT1B','CPT2','ACADVL','ACADM','ACADS','HADHA','HADHB','ECHS1','ACAA2'],
    'Fatty_Acid_Synthesis': ['ACACA','ACACB','FASN','ACLY','ACSS2'],
    'Pentose_Phosphate_Pathway': ['G6PD','PGD','RPIA','RPE','TKTL1','TKT','TALDO1']
}

def get_gene_pathway(gene_name):
    for pathway, genes in PATHWAY_GENES.items():
        if gene_name in genes:
            return pathway
    return "Other"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AMP = torch.cuda.is_available()
if not torch.cuda.is_available():
    torch.set_num_threads(4)
else:
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
print(f"Device: {device}")

class MetaboNicheModel(nn.Module):
    def __init__(self, n_genes=2000, gene_hidden1=1024, gene_hidden2=512, embedding_dim=256, n_pathways=54):
        super().__init__()
        resnet = models.resnet50(pretrained=False)
        self.img_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.img_proj = nn.Linear(2048, embedding_dim)
        self.gene_encoder = nn.Sequential(
            nn.Linear(n_genes, gene_hidden1), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(gene_hidden1, gene_hidden2)
        )
        self.gene_proj = nn.Linear(gene_hidden2, embedding_dim)
        self.fusion = nn.Linear(embedding_dim*2, embedding_dim)
        self.pathway_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(embedding_dim, n_pathways)
        )
        self.gene_decoder = nn.Linear(embedding_dim, n_genes)
    def encode_genes(self, gene_batch):
        return self.gene_proj(self.gene_encoder(gene_batch))
    def encode_image_to_emb(self, image_tensor_1):
        feats = self.img_encoder(image_tensor_1).view(1, -1)
        return self.img_proj(feats)
    def fuse_and_predict(self, gene_emb_batch, img_emb_1):
        img_exp = img_emb_1.expand(gene_emb_batch.shape[0], -1)
        fused = self.fusion(torch.cat([gene_emb_batch, img_exp], dim=1))
        return self.pathway_head(fused)

if GRAPH_AVAILABLE:
    class GraphTransformer(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, heads=4, dropout=0.1):
            super().__init__()
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()
            self.convs.append(TransformerConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden_channels * heads))
            for _ in range(num_layers - 2):
                self.convs.append(TransformerConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
                self.norms.append(nn.LayerNorm(hidden_channels * heads))
            self.convs.append(TransformerConv(hidden_channels * heads, out_channels, heads=1, dropout=dropout))
            self.dropout = dropout
        def forward(self, x, edge_index):
            for conv, norm in zip(self.convs[:-1], self.norms):
                x = conv(x, edge_index)
                x = norm(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x, edge_index)
            return x

def load_model_and_vocab(logger):
    logger.log("Loading models...")
    t0 = time.time()
    try:
        ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        sd = ckpt['model_state_dict']
        g0_w = sd['gene_encoder.0.weight']
        gene_hidden1 = g0_w.shape[0]
        g3_w = sd['gene_encoder.3.weight']
        gene_hidden2 = g3_w.shape[0]
        gp_w = sd.get('gene_proj.weight', sd.get('gene_proj.0.weight'))
        embedding_dim = gp_w.shape[0]
        model = MetaboNicheModel(
            n_genes=2000,
            gene_hidden1=gene_hidden1,
            gene_hidden2=gene_hidden2,
            embedding_dim=embedding_dim,
            n_pathways=len(PATHWAYS)
        ).to(device)
        msd = model.state_dict()
        filtered = {
            k: v for k, v in sd.items()
            if (k in msd and isinstance(v, torch.Tensor) and v.shape == msd[k].shape)
        }
        model.load_state_dict(filtered, strict=False)
        model.eval()
        logger.log(f"Multimodal model loaded in {time.time()-t0:.2f}s")
        with open(VOCAB_PATH, 'r') as f:
            metabolic_genes = json.load(f)
        logger.log(f"Loaded {len(metabolic_genes)} genes")

        graph_model = None
        if GRAPH_AVAILABLE and GRAPH_PATH.exists():
            try:
                graph_ckpt = torch.load(GRAPH_PATH, map_location=device, weights_only=False)
                sample_key = next(
                    k for k in graph_ckpt['model_state_dict'].keys()
                    if 'convs.0.lin_query.weight' in k
                )
                in_channels = graph_ckpt['model_state_dict'][sample_key].shape[1]
                graph_model = GraphTransformer(
                    in_channels=in_channels,
                    hidden_channels=256,
                    out_channels=128
                ).to(device)
                graph_model.load_state_dict(graph_ckpt['model_state_dict'])
                graph_model.eval()
                logger.log("Graph Transformer loaded!")
            except Exception as e:
                logger.log(f"Could not load Graph Transformer: {e}", "WARNING")

        return model, metabolic_genes, graph_model
    except Exception as e:
        logger.error(f"Failed to load: {e}", exc=e)
        raise

startup_logger = Logger()
model, metabolic_genes, graph_model = load_model_and_vocab(startup_logger)
image_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def annotate_cell_types(adata, logger):
    t0 = time.time()
    try:
        if 'leiden_0_6' not in adata.obs.columns:
            sc.pp.neighbors(adata, n_neighbors=15, use_rep='X')
            try:
                sc.tl.leiden(adata, resolution=0.6, key_added='leiden_0_6')
            except ImportError:
                sc.tl.louvain(adata, resolution=0.6, key_added='leiden_0_6')
        cluster_celltype_scores = {}
        for cluster_id in adata.obs['leiden_0_6'].unique():
            cluster_data = adata[adata.obs['leiden_0_6'] == cluster_id]
            celltype_scores = {}
            for celltype, markers in CELL_TYPE_MARKERS.items():
                available_markers = [g for g in markers if g in adata.var_names]
                if len(available_markers) == 0:
                    celltype_scores[celltype] = 0.0
                    continue
                marker_expr = cluster_data[:, available_markers].X
                if sparse.issparse(marker_expr):
                    marker_expr = marker_expr.toarray()
                celltype_scores[celltype] = np.mean(marker_expr)
            best_celltype = max(celltype_scores, key=celltype_scores.get)
            cluster_celltype_scores[cluster_id] = best_celltype
        adata.obs['cell_type'] = adata.obs['leiden_0_6'].map(cluster_celltype_scores)
        logger.log(f"Cell types annotated in {time.time()-t0:.2f}s")
        return adata
    except Exception as e:
        logger.error(f"Cell type annotation failed: {e}", exc=e)
        raise

def _normalize_file_arg(f):
    if f is None:
        return None
    if isinstance(f, str):
        return f
    if isinstance(f, dict):
        return f.get('path') or f.get('name')
    return getattr(f, 'name', None)

def on_file_change(_file):
    if _file is None:
        return "Waiting..."
    if isinstance(_file, dict):
        fname = Path(_file.get('name', 'uploaded')).name
        size = _file.get('size', 0)
    elif isinstance(_file, str):
        p = Path(_file)
        fname = p.name
        size = p.stat().st_size if p.exists() else 0
    else:
        fname = Path(getattr(_file, 'name', 'uploaded')).name
        try:
            size = Path(_file.name).stat().st_size
        except Exception:
            size = 0
    return f"✓ {fname} ({size/(1024*1024):.1f} MB)"

def load_adata_from_extracted(extract_dir, logger):
    """Load AnnData from extracted archive with dynamic file discovery."""
    extract_dir = Path(extract_dir)

    matrix_file = find_file(extract_dir, MATRIX_NAMES)
    features_file = find_file(extract_dir, FEATURES_NAMES)
    barcodes_file = find_file(extract_dir, BARCODES_NAMES)

    if not matrix_file:
        raise FileNotFoundError(f"Could not find matrix file in {extract_dir}. Searched for: {MATRIX_NAMES}")
    if not features_file:
        raise FileNotFoundError(f"Could not find features file in {extract_dir}. Searched for: {FEATURES_NAMES}")
    if not barcodes_file:
        raise FileNotFoundError(f"Could not find barcodes file in {extract_dir}. Searched for: {BARCODES_NAMES}")

    logger.log(f"Found matrix: {matrix_file}")
    logger.log(f"Found features: {features_file}")
    logger.log(f"Found barcodes: {barcodes_file}")

    matrix = read_mtx_file(matrix_file).T.tocsr()
    features = read_tsv_file(features_file)
    barcodes = read_tsv_file(barcodes_file)

    if features.shape[1] >= 2:
        gene_names = features.iloc[:, 1].values
    else:
        gene_names = features.iloc[:, 0].values

    barcode_names = barcodes.iloc[:, 0].values

    adata = ad.AnnData(X=matrix)
    adata.var_names = gene_names
    adata.obs_names = barcode_names
    adata.var_names_make_unique()

    logger.log(f"Loaded count matrix: {adata.n_obs} spots x {adata.n_vars} genes")
    return adata

def load_spatial_data(adata, spatial_dir, logger):
    """Load spatial coordinates and images."""
    spatial_dir = Path(spatial_dir)

    positions_file = find_file(spatial_dir, POSITIONS_NAMES)
    if not positions_file:
        raise FileNotFoundError(f"Could not find positions file in {spatial_dir}. Searched for: {POSITIONS_NAMES}")

    logger.log(f"Found positions: {positions_file}")

    try:
        pos = pd.read_csv(positions_file, header=None, index_col=0)
        if pos.shape[1] >= 5:
            pos.columns = ['in_tissue','array_row','array_col','pxl_row_in_fullres','pxl_col_in_fullres'][:pos.shape[1]]
        elif pos.shape[1] == 4:
            pos.columns = ['in_tissue','array_row','pxl_row_in_fullres','pxl_col_in_fullres']
        else:
            pos.columns = [f'col_{i}' for i in range(pos.shape[1])]
            pos['in_tissue'] = 1
            if 'col_0' in pos.columns:
                pos['pxl_row_in_fullres'] = pos['col_0']
                pos['pxl_col_in_fullres'] = pos['col_1'] if 'col_1' in pos.columns else pos['col_0']
    except Exception as e:
        logger.log(f"Trying alternative positions format: {e}", "WARNING")
        pos = pd.read_csv(positions_file, index_col=0)
        if 'in_tissue' not in pos.columns:
            pos['in_tissue'] = 1

    common_barcodes = adata.obs_names.intersection(pos.index)
    if len(common_barcodes) == 0:
        logger.log("No matching barcodes - using all spots", "WARNING")
        pos.index = adata.obs_names[:len(pos)]
        common_barcodes = adata.obs_names[:len(pos)]

    adata = adata[common_barcodes].copy()
    adata.obs = adata.obs.join(pos)

    if 'in_tissue' in adata.obs.columns:
        in_tissue_mask = adata.obs['in_tissue'] == 1
        if in_tissue_mask.sum() > 0:
            adata = adata[in_tissue_mask].copy()

    logger.log(f"After spatial filtering: {adata.n_obs} spots")

    hires = find_file(spatial_dir, HIRES_IMAGE_NAMES)
    lowres = find_file(spatial_dir, LOWRES_IMAGE_NAMES)
    img_path = hires if hires else lowres

    if img_path:
        logger.log(f"Found image: {img_path}")
    else:
        logger.log("No tissue image found", "WARNING")

    return adata, img_path

def load_single_sample(count_path, spatial_path, meta_path, sample_name, temp_dir, logger):
    """Load a single sample from archives."""
    sample_temp = temp_dir / sample_name
    sample_temp.mkdir(exist_ok=True)

    count_extract = sample_temp / "count"
    spatial_extract = sample_temp / "spatial"
    count_extract.mkdir(exist_ok=True)
    spatial_extract.mkdir(exist_ok=True)

    logger.log(f"Extracting count matrix for {sample_name}...")
    extract_archive(count_path, count_extract)

    logger.log(f"Extracting spatial data for {sample_name}...")
    extract_archive(spatial_path, spatial_extract)

    if meta_path and Path(meta_path).exists():
        meta_extract = sample_temp / "meta"
        meta_extract.mkdir(exist_ok=True)
        extract_archive(meta_path, meta_extract)

    adata = load_adata_from_extracted(count_extract, logger)
    adata, img_path = load_spatial_data(adata, spatial_extract, logger)

    adata.obs['sample'] = sample_name

    return adata, img_path

def build_gene_tensor(adata, mg, logger):
    var_to_idx = {g: i for i, g in enumerate(adata.var_names)}
    present = [i for i, g in enumerate(mg) if g in var_to_idx]
    d = len(mg)
    n = adata.n_obs
    dtype = torch.float16 if AMP else torch.float32
    gene_tensor = torch.zeros((n, d), dtype=dtype, device=device)
    if len(present) > 0:
        X_sub = adata.X[:, [var_to_idx[mg[i]] for i in present]]
        row_bs = 512 if not torch.cuda.is_available() else 2048
        present_set = set(present)
        for s in range(0, n, row_bs):
            e = min(s+row_bs, n)
            block = X_sub[s:e, :]
            if sparse.issparse(block):
                block = block.toarray()
            tmp = np.zeros((e-s, d), dtype=np.float32)
            col = 0
            for j in range(d):
                if j in present_set:
                    tmp[:, j] = block[:, col]
                    col += 1
            gene_tensor[s:e] = torch.from_numpy(tmp).to(device, dtype=dtype)
    return gene_tensor

def compute_shap_values(model, gene_tensor, img_emb_1, adata, metabolic_genes, logger, max_spots=100):
    logger.log("Computing SHAP values...")
    t0 = time.time()
    try:
        n_spots = min(gene_tensor.shape[0], max_spots)
        sampled_indices = np.random.choice(gene_tensor.shape[0], n_spots, replace=False)
        gene_subset = gene_tensor[sampled_indices]
        shap_records = []
        for spot_idx, orig_idx in enumerate(sampled_indices):
            if (spot_idx + 1) % max(1, n_spots // 10) == 0:
                logger.log(f"  SHAP progress: {spot_idx+1}/{n_spots}")
            spot_genes = gene_subset[spot_idx:spot_idx+1]
            spot_genes.requires_grad = True
            gene_emb = model.gene_proj(model.gene_encoder(spot_genes))
            preds = model.fuse_and_predict(gene_emb, img_emb_1)
            spot_id = adata.obs_names[orig_idx]
            cell_type = str(adata.obs.loc[spot_id, 'cell_type']) if 'cell_type' in adata.obs.columns else 'Unknown'
            sample_name = str(adata.obs.loc[spot_id, 'sample']) if 'sample' in adata.obs.columns else 'Unknown'
            for pathway_idx, pathway in enumerate(PATHWAYS):
                pathway_pred = preds[0, pathway_idx]
                model.zero_grad()
                pathway_pred.backward(retain_graph=True)
                gradients = spot_genes.grad[0].cpu().numpy()
                input_values = spot_genes[0].detach().cpu().numpy()
                gene_importance = gradients * input_values
                top_k = 20
                top_indices = np.argsort(np.abs(gene_importance))[-top_k:][::-1]
                record = {
                    'spot_id': spot_id,
                    'sample': sample_name,
                    'cell_type': cell_type,
                    'metabolic_task': pathway,
                    'prediction': float(pathway_pred.detach().cpu().numpy())
                }
                for rank, gene_idx in enumerate(top_indices, 1):
                    gene_name = metabolic_genes[gene_idx] if gene_idx < len(metabolic_genes) else f"Gene_{gene_idx}"
                    gene_pathway = get_gene_pathway(gene_name)
                    record[f'gene_{rank}'] = gene_name
                    record[f'importance_{rank}'] = float(gene_importance[gene_idx])
                    record[f'pathway_{rank}'] = gene_pathway
                shap_records.append(record)
        shap_df = pd.DataFrame(shap_records)
        logger.log(f"SHAP complete in {time.time()-t0:.2f}s")
        return shap_df
    except Exception as e:
        logger.error(f"SHAP failed: {e}", exc=e)
        return pd.DataFrame()

def generate_heatmap_insights(df):
    try:
        insights = []
        insights.append("## 🔍 Key Findings\n")
        n_celltypes = df['cell_type'].nunique()
        n_tissues = df['tissue'].nunique() if 'tissue' in df.columns else 1
        insights.append(f"**Overview:** {n_celltypes} cell types/niches across {n_tissues} tissue(s)\n")
        top_findings = []
        for ct in df['cell_type'].unique()[:5]:
            ct_data = df[df['cell_type'] == ct]
            if not ct_data.empty:
                max_pathway = ct_data.loc[ct_data['trimean'].idxmax()]
                top_findings.append(f"- **{ct}**: Highest in *{max_pathway['metabolic_task']}* ({max_pathway['trimean']:.3f})")
        if top_findings:
            insights.append("\n**Top Activities:**")
            insights.extend(top_findings[:5])
        return "\n".join(insights)
    except Exception as e:
        return f"Error: {str(e)}"

def generate_shap_insights(shap_df):
    try:
        insights = []
        insights.append("## 🧬 Gene Importance\n")
        gene_importance_agg = {}
        gene_pathway_map = {}
        for _, row in shap_df.iterrows():
            for i in range(1, 21):
                gene_col = f'gene_{i}'
                imp_col = f'importance_{i}'
                pathway_col = f'pathway_{i}'
                if gene_col in row and imp_col in row:
                    gene = row[gene_col]
                    imp = row[imp_col]
                    if pd.notna(gene) and pd.notna(imp):
                        gene_importance_agg.setdefault(gene, []).append(abs(imp))
                        if pathway_col in row and pd.notna(row[pathway_col]):
                            gene_pathway_map[gene] = row[pathway_col]
        if not gene_importance_agg:
            return "No SHAP data available."
        avg_importance = {gene: np.mean(imps) for gene, imps in gene_importance_agg.items()}
        top_genes = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        insights.append("**Top 10 Genes:**\n")
        for rank, (gene, imp) in enumerate(top_genes, 1):
            pathway_info = f" ({gene_pathway_map.get(gene, 'Other')})" if gene in gene_pathway_map else ""
            insights.append(f"{rank}. **{gene}**{pathway_info} - {imp:.4f}")
        return "\n".join(insights)
    except Exception as e:
        return f"Error: {str(e)}"

def make_melted(preds, adata, sample_name, logger):
    """Create melted dataframe for a single sample."""
    rows = []
    for i, spot_id in enumerate(adata.obs_names):
        cell_type = str(adata.obs.loc[spot_id, 'cell_type']) if 'cell_type' in adata.obs.columns else 'Unknown'
        cluster = str(adata.obs.loc[spot_id, 'leiden_0_6']) if 'leiden_0_6' in adata.obs.columns else 'Unknown'
        for j, task in enumerate(PATHWAYS):
            rows.append({
                'spot_id': str(spot_id),
                'tissue': sample_name,
                'cell_type': cell_type,
                'cluster': cluster,
                'metabolic_task': task,
                'predicted_score': float(preds[i, j])
            })
    return pd.DataFrame(rows)

def aggregate_predictions(per_spot_df, logger):
    """Aggregate per-spot predictions to cell type level."""
    aggregated_rows = []
    for tissue in per_spot_df['tissue'].unique():
        tissue_data = per_spot_df[per_spot_df['tissue'] == tissue]
        for cell_type in tissue_data['cell_type'].unique():
            for task in PATHWAYS:
                mask = (tissue_data['cell_type'] == cell_type) & (tissue_data['metabolic_task'] == task)
                task_values = tissue_data.loc[mask, 'predicted_score'].values
                if len(task_values) == 0:
                    continue
                q1 = float(np.percentile(task_values, 25))
                median = float(np.percentile(task_values, 50))
                q3 = float(np.percentile(task_values, 75))
                trimean = (q1 + 2 * median + q3) / 4
                aggregated_rows.append({
                    'tissue': tissue,
                    'cell_type': cell_type,
                    'metabolic_task': task,
                    'trimean': float(trimean),
                    'mean': float(task_values.mean()),
                    'std': float(task_values.std()),
                    'min': float(task_values.min()),
                    'max': float(task_values.max()),
                    'q1': q1,
                    'median': median,
                    'q3': q3,
                    'n_cells': int(len(task_values))
                })
    agg_df = pd.DataFrame(aggregated_rows)

    agg_df['scaled_trimean'] = agg_df['trimean']
    for tissue in agg_df['tissue'].unique():
        tissue_mask = agg_df['tissue'] == tissue
        for task in PATHWAYS:
            mask = tissue_mask & (agg_df['metabolic_task'] == task)
            if mask.sum() == 0:
                continue
            task_trimeans = agg_df.loc[mask, 'trimean'].values
            mean_val = task_trimeans.mean()
            std_val = task_trimeans.std()
            if std_val > 0:
                scaled = (task_trimeans - mean_val) / std_val
                agg_df.loc[mask, 'scaled_trimean'] = scaled

    return agg_df

def make_minmax_transposed_from_preds(all_preds, dest, logger):
    """Create min-max CSV from combined predictions."""
    combined = np.vstack(all_preds) if isinstance(all_preds, list) else all_preds
    single_min = combined.min(axis=0)
    single_max = combined.max(axis=0)
    data = {"Metric": ["single_cell_min","single_cell_max","cell_type_min","cell_type_max"]}
    for t_idx, task in enumerate(PATHWAYS):
        data[task] = [
            float(single_min[t_idx]), float(single_max[t_idx]),
            float(single_min[t_idx]), float(single_max[t_idx])
        ]
    pd.DataFrame(data).to_csv(dest, index=False)

def make_spatial_csv(adata, per_spot_df, dest, logger, pathway_name='Glycolysis'):
    """Create spatial CSV with coordinates and pathway scores for ALL pathways."""
    t0 = time.time()
    logger.log(f"Creating comprehensive spatial CSV...")

    rows = []
    for idx, spot_id in enumerate(adata.obs_names):
        row = adata.obs.iloc[idx]
        cell_type = str(row['cell_type']) if 'cell_type' in adata.obs.columns else 'Unknown'
        tissue = str(row['sample']) if 'sample' in adata.obs.columns else 'Unknown'
        x_coord = float(row['pxl_col_in_fullres'])
        y_coord = float(row['pxl_row_in_fullres'])
        niche = str(row['niche']) if 'niche' in adata.obs.columns else 'N/A'

        # Get all pathway scores for this spot
        spot_data = per_spot_df[per_spot_df['spot_id'] == spot_id]

        row_dict = {
            'spot_id': str(spot_id),
            'tissue': tissue,
            'cell_type': cell_type,
            'niche': niche,
            'x_coordinate': x_coord,
            'y_coordinate': y_coord,
        }

        # Add all pathway scores as columns
        for pathway in PATHWAYS:
            pathway_row = spot_data[spot_data['metabolic_task'] == pathway]
            score = float(pathway_row['predicted_score'].iloc[0]) if not pathway_row.empty else 0.0
            row_dict[pathway] = score

        rows.append(row_dict)

    spatial_df = pd.DataFrame(rows)
    spatial_df.to_csv(dest, index=False)
    logger.log(f"Spatial CSV created in {time.time()-t0:.2f}s with {len(PATHWAYS)} pathway columns")
    return spatial_df

def get_tissue_images(tissues_filter):
    """Get tissue images for display."""
    images = []
    if tissues_filter and 'All' not in tissues_filter:
        tissues_to_show = tissues_filter
    else:
        # Show all available images
        tissues_to_show = [f.stem for f in IMAGES_DIR.glob("*.png")]

    for tissue in tissues_to_show:
        img_path = IMAGES_DIR / f"{tissue}.png"
        if img_path.exists():
            images.append((str(img_path), tissue))  # (path, label)

    return images

def get_tissue_celltype_labels(spatial_df):
    """Get tissue names with their actual cell types in brackets."""
    if spatial_df is None or spatial_df.empty:
        return ['All']
    
    tissue_labels = ['All']
    
    # Determine which column to use for cell types
    ct_column = 'niche' if 'niche' in spatial_df.columns else 'cell_type'
    
    for tissue in sorted(spatial_df['tissue'].unique()):
        tissue_data = spatial_df[spatial_df['tissue'] == tissue]
        cell_types = tissue_data[ct_column].dropna().unique()
        
        # Shorten cell type names for display
        ct_short = []
        for ct in cell_types[:3]:  # First 3
            # Take first part before underscore, max 8 chars
            short = str(ct).split('_')[0][:8]
            ct_short.append(short)
        
        if len(cell_types) > 3:
            ct_short.append(f"+{len(cell_types)-3}")
        
        ct_str = ", ".join(ct_short)
        tissue_labels.append(f"{tissue} ({ct_str})")
    
    return tissue_labels

def update_celltype_dropdown(tissues_selected, spatial_df):
    """Update cell type dropdown based on selected tissues."""
    if spatial_df is None or spatial_df.empty:
        return gr.update(choices=['All'], value=['All'])
    
    df = spatial_df.copy()
    
    # Parse tissue names (remove cell type info in brackets) and filter
    if tissues_selected and 'All' not in tissues_selected:
        tissue_names = [t.split(' (')[0] for t in tissues_selected]
        df = df[df['tissue'].isin(tissue_names)]
    
    if len(df) == 0:
        return gr.update(choices=['All'], value=['All'])
    
    # Get cell types from filtered data - use 'niche' if available (for graph), else 'cell_type'
    if 'niche' in df.columns:
        unique_celltypes = sorted(df['niche'].dropna().unique().tolist())
    else:
        unique_celltypes = sorted(df['cell_type'].dropna().unique().tolist())
    
    cell_types = ['All'] + unique_celltypes
    return gr.update(choices=cell_types, value=['All'])

def create_tissue_image_gallery(tissues_filter):
    """Create a gallery of tissue images with labels."""
    images = get_tissue_images(tissues_filter)
    if not images:
        return []
    # Return tuples of (image_path, label) for Gallery
    return [(img[0], img[1]) for img in images]

def create_heatmap(df_filtered, use_scaled, color_palette='RdBu_r'):
    try:
        if df_filtered.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data", showarrow=False)
            return fig
        value_col = 'scaled_trimean' if use_scaled else 'trimean'
        pivot_df = df_filtered.pivot_table(
            index='cell_type', columns='metabolic_task',
            values=value_col, aggfunc='mean'
        )
        hover_text = []
        for cell_type in pivot_df.index:
            row_text = []
            for task in pivot_df.columns:
                cell_data = df_filtered[
                    (df_filtered['cell_type'] == cell_type) &
                    (df_filtered['metabolic_task'] == task)
                ]
                if not cell_data.empty:
                    row = cell_data.iloc[0]
                    text = f"<b>{cell_type}</b><br><b>{task}</b><br><br>"
                    text += f"Trimean: {row['trimean']:.4f}<br>"
                    text += f"Scaled: {row['scaled_trimean']:.4f}<br>"
                    text += f"Mean: {row['mean']:.4f}<br>"
                    text += f"Std: {row['std']:.4f}<br>"
                    text += f"Range: [{row['min']:.4f}, {row['max']:.4f}]<br>"
                    text += f"N Cells: {int(row['n_cells'])}"
                else:
                    text = "No data"
                row_text.append(text)
            hover_text.append(row_text)
        n_cell_types = len(pivot_df.index)
        n_pathways = len(pivot_df.columns)
        plot_height = max(600, n_cell_types * 40 + 200)
        plot_width = max(1400, n_pathways * 60 + 300)
        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale=color_palette,
            zmid=0 if use_scaled else None,
            colorbar=dict(title="Scaled" if use_scaled else "Trimean", len=0.4),
            hovertemplate='%{text}<extra></extra>',
            text=hover_text
        ))
        fig.update_layout(
            title=f"Metabolic Heatmap: {n_cell_types} Cell Types × {n_pathways} Pathways",
            xaxis=dict(title="Metabolic Pathway", tickangle=-45, tickfont=dict(size=11)),
            yaxis=dict(title="Cell Type", tickfont=dict(size=12)),
            height=plot_height,
            width=plot_width,
            margin=dict(l=220, r=150, t=120, b=280),
            paper_bgcolor='white',
            plot_bgcolor='#fafafa'
        )
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", showarrow=False, font=dict(color='red'))
        return fig

def create_shap_plot(shap_path, top_n=20):
    try:
        df = pd.read_csv(shap_path)
        gene_importance_agg = {}
        gene_pathway_map = {}
        for _, row in df.iterrows():
            for i in range(1, 21):
                gene_col = f'gene_{i}'
                imp_col = f'importance_{i}'
                pathway_col = f'pathway_{i}'
                if gene_col in row and imp_col in row:
                    gene = row[gene_col]
                    imp = row[imp_col]
                    if pd.notna(gene) and pd.notna(imp):
                        gene_importance_agg.setdefault(gene, []).append(abs(imp))
                        if pathway_col in row and pd.notna(row[pathway_col]):
                            gene_pathway_map[gene] = row[pathway_col]
        if not gene_importance_agg:
            fig = go.Figure()
            fig.add_annotation(text="No SHAP data", showarrow=False)
            return fig
        avg_importance = {gene: np.mean(imps) for gene, imps in gene_importance_agg.items()}
        top_genes = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        genes, importances = zip(*top_genes)
        gene_labels = [f"{g} ({gene_pathway_map.get(g, 'Other')})" for g in genes]
        plot_height = max(600, top_n * 28)
        fig = go.Figure(data=[
            go.Bar(
                x=list(importances),
                y=gene_labels,
                orientation='h',
                marker=dict(color=list(importances), colorscale='Viridis', showscale=False),
                hovertemplate='<b>%{y}</b><br>Importance: %{x:.6f}<extra></extra>'
            )
        ])
        fig.update_layout(
            title=f"Top {top_n} Most Important Genes",
            xaxis_title="Average Absolute Importance",
            yaxis_title="Gene (Pathway)",
            height=plot_height,
            width=1000,
            yaxis={'categoryorder': 'total ascending'},
            paper_bgcolor='white',
            plot_bgcolor='#fafafa',
            margin=dict(l=200, r=100, t=100, b=100)
        )
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", showarrow=False, font=dict(color='red'))
        return fig
def create_spatial_statistics_plots(spatial_df, tissues_filter, celltypes_filter, pathway_name='Glycolysis', color_palette='jet'):
    """Create statistics plots: Intensity distribution, Cell type composition, Spots per tissue."""
    try:
        if spatial_df is None or spatial_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No spatial data available", showarrow=False)
            return fig

        df = spatial_df.copy()

        # Parse tissue names (remove cell type info in brackets)
        if tissues_filter and len(tissues_filter) > 0 and 'All' not in tissues_filter:
            tissue_names = [t.split(' (')[0] for t in tissues_filter]
            df = df[df['tissue'].isin(tissue_names)]

        # Apply cell type filter
        if celltypes_filter and len(celltypes_filter) > 0 and 'All' not in celltypes_filter:
            df = df[df['cell_type'].isin(celltypes_filter)]

        if len(df) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No spots match filters", showarrow=False)
            return fig

        cell_types = df['cell_type'].values
        tissues = df['tissue'].values

        # Get pathway scores
        if pathway_name in df.columns:
            pathway_scores = df[pathway_name].values
        else:
            pathway_scores = np.zeros(len(df))

        # Color maps
        unique_celltypes = np.unique(cell_types)
        colors = px.colors.qualitative.Set2 + px.colors.qualitative.Set1 + px.colors.qualitative.Pastel1
        ct_color_map = {ct: colors[i % len(colors)] for i, ct in enumerate(unique_celltypes)}

        unique_tissues = np.unique(tissues)
        tissue_colors = px.colors.qualitative.Plotly + px.colors.qualitative.D3
        tissue_color_map = {t: tissue_colors[i % len(tissue_colors)] for i, t in enumerate(unique_tissues)}

        # Create 1x3 subplot layout
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                f'<b>{pathway_name}</b> Intensity Distribution',
                '<b>Cell Type Composition</b>',
                '<b>Spots per Tissue</b>'
            ),
            specs=[[{'type': 'bar'}, {'type': 'pie'}, {'type': 'bar'}]],
            horizontal_spacing=0.1
        )

        # Panel 1: Pathway intensity distribution (histogram as bar)
        hist_values, hist_edges = np.histogram(pathway_scores, bins=20)
        hist_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
        fig.add_trace(
            go.Bar(
                x=hist_centers,
                y=hist_values,
                marker=dict(
                    color=hist_centers,
                    colorscale=color_palette,
                    showscale=False
                ),
                hovertemplate='Score: %{x:.2f}<br>Count: %{y}<extra></extra>',
                showlegend=False
            ),
            row=1, col=1
        )

        # Panel 2: Cell type composition (pie chart)
        ct_counts = pd.Series(cell_types).value_counts()
        fig.add_trace(
            go.Pie(
                labels=ct_counts.index.tolist(),
                values=ct_counts.values.tolist(),
                marker=dict(colors=[ct_color_map[ct] for ct in ct_counts.index]),
                textinfo='percent+label',
                textposition='inside',
                insidetextorientation='radial',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>',
                showlegend=False
            ),
            row=1, col=2
        )

        # Panel 3: Spots per tissue (bar chart)
        tissue_counts = pd.Series(tissues).value_counts()
        fig.add_trace(
            go.Bar(
                x=tissue_counts.index.tolist(),
                y=tissue_counts.values.tolist(),
                marker=dict(color=[tissue_color_map[t] for t in tissue_counts.index]),
                hovertemplate='<b>%{x}</b><br>Spots: %{y}<extra></extra>',
                showlegend=False
            ),
            row=1, col=3
        )

        # Add axis labels
        fig.update_xaxes(title_text="Flux Score", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_xaxes(title_text="Tissue", tickangle=-45, row=1, col=3)
        fig.update_yaxes(title_text="Spots", row=1, col=3)

        fig.update_layout(
            height=350,
            width=1600,
            showlegend=False,
            paper_bgcolor='white',
            plot_bgcolor='#fafafa',
            margin=dict(l=60, r=60, t=60, b=80)
        )

        return fig

    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", showarrow=False, font=dict(color='red'))
        return fig

def create_spatial_plots_from_csv(spatial_df, tissues_filter, celltypes_filter, pathway_name='Glycolysis', 
                                   color_palette='jet', point_size=6):
    """Create spatial plots: Tissue image (if single), Predicted pathway, Ground Truth cell types, Tissue samples."""
    try:
        if spatial_df is None or spatial_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No spatial data available", showarrow=False)
            return fig
        
        df = spatial_df.copy()
        
        # Parse tissue names (remove cell type info in brackets)
        tissue_names_parsed = None
        if tissues_filter and len(tissues_filter) > 0 and 'All' not in tissues_filter:
            tissue_names_parsed = [t.split(' (')[0] for t in tissues_filter]
            df = df[df['tissue'].isin(tissue_names_parsed)]
        
        # Apply cell type filter
        if celltypes_filter and len(celltypes_filter) > 0 and 'All' not in celltypes_filter:
            df = df[df['cell_type'].isin(celltypes_filter)]
        
        if len(df) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No spots match filters", showarrow=False)
            return fig
        
        x = df['x_coordinate'].values
        y = df['y_coordinate'].values
        cell_types = df['cell_type'].values
        tissues = df['tissue'].values
        
        # Get pathway scores
        if pathway_name in df.columns:
            pathway_scores = df[pathway_name].values
        else:
            pathway_scores = np.zeros(len(x))
        
        # Check if single tissue is selected
        unique_tissues = np.unique(tissues)
        single_tissue = len(unique_tissues) == 1
        
        if single_tissue:
            # 1x4 layout with tissue image on the left
            tissue_name = unique_tissues[0]
            tissue_img_path = IMAGES_DIR / f"{tissue_name}.png"
            
            fig = make_subplots(
                rows=1, cols=4,
                subplot_titles=(
                    f'<b>H&E Image:</b> {tissue_name}',
                    f'<b>Predicted:</b> {pathway_name} Activity',
                    '<b>Ground Truth:</b> Cell Types',
                    '<b>Tissue Sample</b>'
                ),
                specs=[[{'type': 'image'}, {'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}]],
                horizontal_spacing=0.06,
                column_widths=[0.25, 0.25, 0.25, 0.25]
            )
            
            # Panel 1: Tissue H&E Image
            if tissue_img_path.exists():
                img = Image.open(tissue_img_path)
                fig.add_trace(
                    go.Image(z=np.array(img)),
                    row=1, col=1
                )
            else:
                # Placeholder if no image
                fig.add_annotation(
                    text=f"No image for {tissue_name}",
                    xref="x domain", yref="y domain",
                    x=0.5, y=0.5, showarrow=False,
                    row=1, col=1
                )
            
            # Panel 2: Predicted Pathway Activity
            fig.add_trace(
                go.Scatter(
                    x=x, y=y,
                    mode='markers',
                    marker=dict(
                        size=point_size,
                        color=pathway_scores,
                        colorscale=color_palette,
                        showscale=True,
                        colorbar=dict(
                            title=dict(text="Flux", side="right"),
                            x=0.52,
                            len=0.4,
                            y=0.5
                        ),
                        line=dict(width=0.5, color='white')
                    ),
                    hovertemplate=f'<b>{pathway_name}</b><br>Score: %{{marker.color:.3f}}<br>X: %{{x}}<br>Y: %{{y}}<extra></extra>',
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # Panel 3: Ground Truth Cell Types
            unique_celltypes = np.unique(cell_types)
            colors = px.colors.qualitative.Set2 + px.colors.qualitative.Set1 + px.colors.qualitative.Pastel1
            ct_color_map = {ct: colors[i % len(colors)] for i, ct in enumerate(unique_celltypes)}
            
            for ct in unique_celltypes:
                mask = cell_types == ct
                fig.add_trace(
                    go.Scatter(
                        x=x[mask], y=y[mask],
                        mode='markers',
                        name=ct,
                        marker=dict(
                            size=point_size,
                            color=ct_color_map[ct],
                            opacity=0.8,
                            line=dict(width=0.5, color='white')
                        ),
                        hovertemplate=f'<b>{ct}</b><br>X: %{{x}}<br>Y: %{{y}}<extra></extra>',
                        legendgroup=ct
                    ),
                    row=1, col=3
                )
            
            # Panel 4: Tissue Sample (single color)
            tissue_colors = px.colors.qualitative.Plotly
            fig.add_trace(
                go.Scatter(
                    x=x, y=y,
                    mode='markers',
                    name=tissue_name,
                    marker=dict(
                        size=point_size,
                        color=tissue_colors[0],
                        opacity=0.8,
                        line=dict(width=0.5, color='white')
                    ),
                    hovertemplate=f'<b>{tissue_name}</b><br>X: %{{x}}<br>Y: %{{y}}<extra></extra>',
                    showlegend=False
                ),
                row=1, col=4
            )
            
            # Update axes for spatial plots (cols 2, 3, 4)
            for col in [2, 3, 4]:
                fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=col)
                fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, autorange='reversed', row=1, col=col)
            
            # Hide axes for image
            fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=1)
            fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=1)
            
            fig.update_layout(
                height=500,
                width=1800,
                title_text=f"Spatial Metabolic Analysis: {tissue_name} - {pathway_name}",
                showlegend=True,
                legend=dict(
                    x=1.02,
                    y=0.5,
                    xanchor='left',
                    yanchor='middle',
                    title=dict(text='Cell Types'),
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='#ccc',
                    borderwidth=1
                ),
                paper_bgcolor='white',
                plot_bgcolor='#fafafa',
                margin=dict(l=60, r=200, t=80, b=60)
            )
        
        else:
            # Multiple tissues - original 1x3 layout without image
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=(
                    f'<b>Predicted:</b> {pathway_name} Activity',
                    '<b>Ground Truth:</b> Cell Types',
                    '<b>Tissue Samples</b>'
                ),
                specs=[[{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}]],
                horizontal_spacing=0.08
            )
            
            # Panel 1: Predicted Pathway Activity
            fig.add_trace(
                go.Scatter(
                    x=x, y=y,
                    mode='markers',
                    marker=dict(
                        size=point_size,
                        color=pathway_scores,
                        colorscale=color_palette,
                        showscale=True,
                        colorbar=dict(
                            title=dict(text="Flux", side="right"),
                            x=0.28,
                            len=0.4,
                            y=0.5
                        ),
                        line=dict(width=0.5, color='white')
                    ),
                    hovertemplate=f'<b>{pathway_name}</b><br>Score: %{{marker.color:.3f}}<br>X: %{{x}}<br>Y: %{{y}}<extra></extra>',
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Panel 2: Ground Truth Cell Types
            unique_celltypes = np.unique(cell_types)
            colors = px.colors.qualitative.Set2 + px.colors.qualitative.Set1 + px.colors.qualitative.Pastel1
            ct_color_map = {ct: colors[i % len(colors)] for i, ct in enumerate(unique_celltypes)}
            
            for ct in unique_celltypes:
                mask = cell_types == ct
                fig.add_trace(
                    go.Scatter(
                        x=x[mask], y=y[mask],
                        mode='markers',
                        name=ct,
                        marker=dict(
                            size=point_size,
                            color=ct_color_map[ct],
                            opacity=0.8,
                            line=dict(width=0.5, color='white')
                        ),
                        hovertemplate=f'<b>{ct}</b><br>X: %{{x}}<br>Y: %{{y}}<extra></extra>',
                        legendgroup=ct
                    ),
                    row=1, col=2
                )
            
            # Panel 3: Tissue Samples
            tissue_colors = px.colors.qualitative.Plotly + px.colors.qualitative.D3
            tissue_color_map = {t: tissue_colors[i % len(tissue_colors)] for i, t in enumerate(unique_tissues)}
            
            for tissue in unique_tissues:
                mask = tissues == tissue
                fig.add_trace(
                    go.Scatter(
                        x=x[mask], y=y[mask],
                        mode='markers',
                        name=tissue,
                        marker=dict(
                            size=point_size,
                            color=tissue_color_map[tissue],
                            opacity=0.8,
                            line=dict(width=0.5, color='white')
                        ),
                        hovertemplate=f'<b>{tissue}</b><br>X: %{{x}}<br>Y: %{{y}}<extra></extra>',
                        showlegend=False
                    ),
                    row=1, col=3
                )
            
            # Update axes
            for col in [1, 2, 3]:
                fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=col)
                fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, autorange='reversed', row=1, col=col)
            
            fig.update_layout(
                height=500,
                width=1600,
                title_text=f"Spatial Metabolic Analysis: {pathway_name}",
                showlegend=True,
                legend=dict(
                    x=1.02,
                    y=0.5,
                    xanchor='left',
                    yanchor='middle',
                    title=dict(text='Cell Types'),
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='#ccc',
                    borderwidth=1
                ),
                paper_bgcolor='white',
                plot_bgcolor='#fafafa',
                margin=dict(l=60, r=200, t=80, b=60)
            )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", showarrow=False, font=dict(color='red'))
        return fig

def create_niche_statistics_plots(spatial_df, tissues_filter, celltypes_filter, pathway_name='Glycolysis', color_palette='jet'):
    """Create statistics plots for niches: Intensity distribution, Niche composition, Spots per tissue."""
    try:
        if spatial_df is None or spatial_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No spatial data available", showarrow=False)
            return fig

        df = spatial_df.copy()

        # Parse tissue names (remove cell type info in brackets)
        if tissues_filter and len(tissues_filter) > 0 and 'All' not in tissues_filter:
            tissue_names = [t.split(' (')[0] for t in tissues_filter]
            df = df[df['tissue'].isin(tissue_names)]

        # Apply niche/cell type filter
        if celltypes_filter and len(celltypes_filter) > 0 and 'All' not in celltypes_filter:
            if 'niche' in df.columns:
                df = df[df['niche'].isin(celltypes_filter)]
            else:
                df = df[df['cell_type'].isin(celltypes_filter)]

        if len(df) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No spots match filters", showarrow=False)
            return fig

        niches = df['niche'].values if 'niche' in df.columns else df['cell_type'].values
        tissues = df['tissue'].values

        # Get pathway scores
        if pathway_name in df.columns:
            pathway_scores = df[pathway_name].values
        else:
            pathway_scores = np.zeros(len(df))

        # Color maps
        unique_niches = np.unique(niches)
        colors = px.colors.qualitative.Set2 + px.colors.qualitative.Set1 + px.colors.qualitative.Pastel1
        niche_color_map = {n: colors[i % len(colors)] for i, n in enumerate(unique_niches)}

        unique_tissues = np.unique(tissues)
        tissue_colors = px.colors.qualitative.Plotly + px.colors.qualitative.D3
        tissue_color_map = {t: tissue_colors[i % len(tissue_colors)] for i, t in enumerate(unique_tissues)}

        # Create 1x3 subplot layout
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                f'<b>{pathway_name}</b> Intensity Distribution',
                '<b>Niche Composition</b>',
                '<b>Spots per Tissue</b>'
            ),
            specs=[[{'type': 'bar'}, {'type': 'pie'}, {'type': 'bar'}]],
            horizontal_spacing=0.1
        )

        # Panel 1: Pathway intensity distribution
        hist_values, hist_edges = np.histogram(pathway_scores, bins=20)
        hist_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
        fig.add_trace(
            go.Bar(
                x=hist_centers,
                y=hist_values,
                marker=dict(
                    color=hist_centers,
                    colorscale=color_palette,
                    showscale=False
                ),
                hovertemplate='Score: %{x:.2f}<br>Count: %{y}<extra></extra>',
                showlegend=False
            ),
            row=1, col=1
        )

        # Panel 2: Niche composition (pie chart)
        niche_counts = pd.Series(niches).value_counts()
        fig.add_trace(
            go.Pie(
                labels=niche_counts.index.tolist(),
                values=niche_counts.values.tolist(),
                marker=dict(colors=[niche_color_map[n] for n in niche_counts.index]),
                textinfo='percent+label',
                textposition='inside',
                insidetextorientation='radial',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>',
                showlegend=False
            ),
            row=1, col=2
        )

        # Panel 3: Spots per tissue
        tissue_counts = pd.Series(tissues).value_counts()
        fig.add_trace(
            go.Bar(
                x=tissue_counts.index.tolist(),
                y=tissue_counts.values.tolist(),
                marker=dict(color=[tissue_color_map[t] for t in tissue_counts.index]),
                hovertemplate='<b>%{x}</b><br>Spots: %{y}<extra></extra>',
                showlegend=False
            ),
            row=1, col=3
        )

        # Add axis labels
        fig.update_xaxes(title_text="Flux Score", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_xaxes(title_text="Tissue", tickangle=-45, row=1, col=3)
        fig.update_yaxes(title_text="Spots", row=1, col=3)

        fig.update_layout(
            height=350,
            width=1600,
            showlegend=False,
            paper_bgcolor='white',
            plot_bgcolor='#fafafa',
            margin=dict(l=60, r=60, t=60, b=80)
        )

        return fig

    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", showarrow=False, font=dict(color='red'))
        return fig
def create_niche_visualization_from_csv(spatial_df, tissues_filter, celltypes_filter, pathway_name='Glycolysis',
                                         color_palette='jet', point_size=6):
    """Create niche visualization - with tissue image if single tissue selected."""
    try:
        if spatial_df is None or spatial_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No spatial data available", showarrow=False)
            return fig
        
        df = spatial_df.copy()
        
        # Parse tissue names (remove cell type info in brackets)
        tissue_names_parsed = None
        if tissues_filter and len(tissues_filter) > 0 and 'All' not in tissues_filter:
            tissue_names_parsed = [t.split(' (')[0] for t in tissues_filter]
            df = df[df['tissue'].isin(tissue_names_parsed)]
        
        # Apply cell type/niche filter
        if celltypes_filter and len(celltypes_filter) > 0 and 'All' not in celltypes_filter:
            if 'niche' in df.columns:
                df = df[df['niche'].isin(celltypes_filter)]
            else:
                df = df[df['cell_type'].isin(celltypes_filter)]
        
        if len(df) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No spots match filters", showarrow=False)
            return fig
        
        x = df['x_coordinate'].values
        y = df['y_coordinate'].values
        cell_types = df['cell_type'].values
        tissues = df['tissue'].values
        niches = df['niche'].values if 'niche' in df.columns else cell_types
        
        # Get pathway scores
        if pathway_name in df.columns:
            pathway_scores = df[pathway_name].values
        else:
            pathway_scores = np.zeros(len(x))
        
        # Check if single tissue is selected
        unique_tissues = np.unique(tissues)
        single_tissue = len(unique_tissues) == 1
        
        if single_tissue:
            # 1x4 layout with tissue image on the left
            tissue_name = unique_tissues[0]
            tissue_img_path = IMAGES_DIR / f"{tissue_name}.png"
            
            fig = make_subplots(
                rows=1, cols=4,
                subplot_titles=(
                    f'<b>H&E Image:</b> {tissue_name}',
                    f'<b>Predicted:</b> {pathway_name} Activity',
                    '<b>Ground Truth:</b> Metabolic Niches',
                    '<b>Tissue Sample</b>'
                ),
                specs=[[{'type': 'image'}, {'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}]],
                horizontal_spacing=0.06,
                column_widths=[0.25, 0.25, 0.25, 0.25]
            )
            
            # Panel 1: Tissue H&E Image
            if tissue_img_path.exists():
                img = Image.open(tissue_img_path)
                fig.add_trace(
                    go.Image(z=np.array(img)),
                    row=1, col=1
                )
            else:
                fig.add_annotation(
                    text=f"No image for {tissue_name}",
                    xref="x domain", yref="y domain",
                    x=0.5, y=0.5, showarrow=False,
                    row=1, col=1
                )
            
            # Panel 2: Predicted Pathway Activity
            fig.add_trace(
                go.Scatter(
                    x=x, y=y,
                    mode='markers',
                    marker=dict(
                        size=point_size,
                        color=pathway_scores,
                        colorscale=color_palette,
                        showscale=True,
                        colorbar=dict(
                            title=dict(text="Flux", side="right"),
                            x=0.52,
                            len=0.4,
                            y=0.5
                        ),
                        line=dict(width=0.5, color='white')
                    ),
                    hovertemplate=f'<b>{pathway_name}</b><br>Score: %{{marker.color:.3f}}<br>X: %{{x}}<br>Y: %{{y}}<extra></extra>',
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # Panel 3: Ground Truth Niches
            unique_niches = np.unique(niches)
            colors = px.colors.qualitative.Set2 + px.colors.qualitative.Set1 + px.colors.qualitative.Pastel1
            niche_color_map = {n: colors[i % len(colors)] for i, n in enumerate(unique_niches)}
            
            for niche in unique_niches:
                mask = niches == niche
                fig.add_trace(
                    go.Scatter(
                        x=x[mask], y=y[mask],
                        mode='markers',
                        name=str(niche),
                        marker=dict(
                            size=point_size,
                            color=niche_color_map[niche],
                            opacity=0.8,
                            line=dict(width=0.5, color='white')
                        ),
                        hovertemplate=f'<b>{niche}</b><br>X: %{{x}}<br>Y: %{{y}}<extra></extra>',
                        legendgroup=str(niche)
                    ),
                    row=1, col=3
                )
            
            # Panel 4: Tissue Sample
            tissue_colors = px.colors.qualitative.Plotly
            fig.add_trace(
                go.Scatter(
                    x=x, y=y,
                    mode='markers',
                    name=tissue_name,
                    marker=dict(
                        size=point_size,
                        color=tissue_colors[0],
                        opacity=0.8,
                        line=dict(width=0.5, color='white')
                    ),
                    hovertemplate=f'<b>{tissue_name}</b><br>X: %{{x}}<br>Y: %{{y}}<extra></extra>',
                    showlegend=False
                ),
                row=1, col=4
            )
            
            # Update axes for spatial plots (cols 2, 3, 4)
            for col in [2, 3, 4]:
                fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=col)
                fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, autorange='reversed', row=1, col=col)
            
            # Hide axes for image
            fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=1)
            fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=1)
            
            fig.update_layout(
                height=500,
                width=1800,
                title_text=f"Spatial Metabolic Niche Analysis: {tissue_name} - {pathway_name}",
                showlegend=True,
                legend=dict(
                    x=1.02,
                    y=0.5,
                    xanchor='left',
                    yanchor='middle',
                    title=dict(text='Niches'),
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='#ccc',
                    borderwidth=1
                ),
                paper_bgcolor='white',
                plot_bgcolor='#fafafa',
                margin=dict(l=60, r=200, t=80, b=60)
            )
        
        else:
            # Multiple tissues - original 1x3 layout without image
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=(
                    f'<b>Predicted:</b> {pathway_name} Activity',
                    '<b>Ground Truth:</b> Metabolic Niches',
                    '<b>Tissue Samples</b>'
                ),
                specs=[[{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}]],
                horizontal_spacing=0.08
            )
            
            # Panel 1: Predicted Pathway Activity
            fig.add_trace(
                go.Scatter(
                    x=x, y=y,
                    mode='markers',
                    marker=dict(
                        size=point_size,
                        color=pathway_scores,
                        colorscale=color_palette,
                        showscale=True,
                        colorbar=dict(
                            title=dict(text="Flux", side="right"),
                            x=0.28,
                            len=0.4,
                            y=0.5
                        ),
                        line=dict(width=0.5, color='white')
                    ),
                    hovertemplate=f'<b>{pathway_name}</b><br>Score: %{{marker.color:.3f}}<br>X: %{{x}}<br>Y: %{{y}}<extra></extra>',
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Panel 2: Ground Truth Niches
            unique_niches = np.unique(niches)
            colors = px.colors.qualitative.Set2 + px.colors.qualitative.Set1 + px.colors.qualitative.Pastel1
            niche_color_map = {n: colors[i % len(colors)] for i, n in enumerate(unique_niches)}
            
            for niche in unique_niches:
                mask = niches == niche
                fig.add_trace(
                    go.Scatter(
                        x=x[mask], y=y[mask],
                        mode='markers',
                        name=str(niche),
                        marker=dict(
                            size=point_size,
                            color=niche_color_map[niche],
                            opacity=0.8,
                            line=dict(width=0.5, color='white')
                        ),
                        hovertemplate=f'<b>{niche}</b><br>X: %{{x}}<br>Y: %{{y}}<extra></extra>',
                        legendgroup=str(niche)
                    ),
                    row=1, col=2
                )
            
            # Panel 3: Tissue Samples
            tissue_colors = px.colors.qualitative.Plotly + px.colors.qualitative.D3
            tissue_color_map = {t: tissue_colors[i % len(tissue_colors)] for i, t in enumerate(unique_tissues)}
            
            for tissue in unique_tissues:
                mask = tissues == tissue
                fig.add_trace(
                    go.Scatter(
                        x=x[mask], y=y[mask],
                        mode='markers',
                        name=tissue,
                        marker=dict(
                            size=point_size,
                            color=tissue_color_map[tissue],
                            opacity=0.8,
                            line=dict(width=0.5, color='white')
                        ),
                        hovertemplate=f'<b>{tissue}</b><br>X: %{{x}}<br>Y: %{{y}}<extra></extra>',
                        showlegend=False
                    ),
                    row=1, col=3
                )
            
            # Update axes
            for col in [1, 2, 3]:
                fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=col)
                fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, autorange='reversed', row=1, col=col)
            
            fig.update_layout(
                height=500,
                width=1600,
                title_text=f"Spatial Metabolic Niche Analysis: {pathway_name}",
                showlegend=True,
                legend=dict(
                    x=1.02,
                    y=0.5,
                    xanchor='left',
                    yanchor='middle',
                    title=dict(text='Niches'),
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='#ccc',
                    borderwidth=1
                ),
                paper_bgcolor='white',
                plot_bgcolor='#fafafa',
                margin=dict(l=60, r=200, t=80, b=60)
            )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", showarrow=False, font=dict(color='red'))
        return fig

# Global storage
mm_stored_melted_df = None
mm_stored_spatial_df = None
mm_stored_heatmap_fig = None
mm_stored_shap_fig = None
mm_stored_spatial_fig = None

graph_stored_melted_df = None
graph_stored_spatial_df = None
graph_stored_heatmap_fig = None
graph_stored_shap_fig = None
graph_stored_spatial_fig = None

def explain_visualization_with_openai(api_key, df, shap_df=None, viz_type="heatmap"):
    if not OPENAI_AVAILABLE:
        return "ERROR: OpenAI library not installed."
    if df is None or df.empty:
        return "ERROR: No data available. Run pipeline first."
    if not api_key or api_key.strip() == "":
        return "ERROR: Please enter your OpenAI API key."

    try:
        client = openai.OpenAI(api_key=api_key.strip())
    except Exception as e:
        return f"ERROR: Failed to initialize OpenAI client: {e}"

    if viz_type == "heatmap":
        top_celltypes = df.groupby('cell_type')['trimean'].mean().nlargest(10)
        top_pathways = df.groupby('metabolic_task')['trimean'].mean().nlargest(10)
        n_tissues = df['tissue'].nunique()

        data_summary = f"""
**Heatmap Data Summary:**
**Tissues analyzed:** {n_tissues} ({', '.join(df['tissue'].unique()[:5])})
**Top 10 Cell Types by Average Metabolic Activity:**
{top_celltypes.to_dict()}
**Top 10 Most Active Pathways:**
{top_pathways.to_dict()}
**Sample of detailed data (first 20 rows):**
{df.head(20).to_dict('records')}
"""
        prompt = f"""You are an expert in spatial transcriptomics and metabolic analysis.
Below is data from a MetaboNiche metabolic heatmap:
{data_summary}
Please provide:
1. Brief overview of the metabolic landscape
2. Key patterns across tissues
3. Potential biological significance
4. Any interesting metabolic relationships
Keep explanation clear and concise."""

    elif viz_type == "shap":
        if shap_df is None or shap_df.empty:
            return "ERROR: No SHAP data available."

        gene_importance = {}
        for _, row in shap_df.iterrows():
            for i in range(1, 21):
                gene_col = f'gene_{i}'
                imp_col = f'importance_{i}'
                if gene_col in row and imp_col in row:
                    gene = row[gene_col]
                    imp = row[imp_col]
                    if pd.notna(gene) and pd.notna(imp):
                        gene_importance.setdefault(gene, []).append(abs(imp))

        avg_importance = {gene: np.mean(imps) for gene, imps in gene_importance.items()}
        top_genes = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:15]

        data_summary = f"""
**SHAP Gene Importance Summary:**
**Top 15 Most Important Genes:**
{dict(top_genes)}
**Statistics:**
- Total spots analyzed: {shap_df['spot_id'].nunique() if 'spot_id' in shap_df.columns else 'N/A'}
- Tissues: {shap_df['sample'].nunique() if 'sample' in shap_df.columns else 'N/A'}
"""
        prompt = f"""You are an expert in spatial transcriptomics.
Below is SHAP gene importance data:
{data_summary}
Please provide:
1. Overview of gene importance patterns
2. Key genes and their biological roles
3. Insights into pathway drivers
4. Potential biological implications
Keep explanation clear and concise."""

    else:
        cell_type_summary = df.groupby('cell_type')['trimean'].agg(['mean', 'std', 'count'])
        tissue_summary = df.groupby('tissue')['trimean'].agg(['mean', 'std', 'count'])

        data_summary = f"""
**Spatial Metabolic Data Summary:**
**Tissue Statistics:**
{tissue_summary.to_dict()}
**Cell Type Statistics:**
{cell_type_summary.head(10).to_dict()}
"""
        prompt = f"""You are an expert in spatial transcriptomics.
Below is spatial metabolic data:
{data_summary}
Please provide:
1. Overview of spatial organization
2. Tissue-specific patterns
3. Cell type metabolic profiles
4. Biological significance
Keep explanation clear and concise."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: OpenAI request failed: {e}"

def explain_mm_heatmap(api_key):
    global mm_stored_melted_df
    return explain_visualization_with_openai(api_key, mm_stored_melted_df, viz_type="heatmap")

def explain_mm_shap(api_key):
    global mm_stored_melted_df
    shap_path = OUT_DIR / "mm_shap.csv"
    if not shap_path.exists():
        shap_path = RESULTS_DIR / "mm_shap.csv"
    shap_df = pd.read_csv(shap_path) if shap_path.exists() else None
    return explain_visualization_with_openai(api_key, mm_stored_melted_df, shap_df=shap_df, viz_type="shap")

def explain_mm_spatial(api_key):
    global mm_stored_melted_df
    return explain_visualization_with_openai(api_key, mm_stored_melted_df, viz_type="spatial")

def explain_graph_heatmap(api_key):
    global graph_stored_melted_df
    return explain_visualization_with_openai(api_key, graph_stored_melted_df, viz_type="heatmap")

def explain_graph_shap(api_key):
    global graph_stored_melted_df
    shap_path = OUT_DIR / "graph_shap.csv"
    if not shap_path.exists():
        shap_path = RESULTS_DIR / "graph_shap.csv"
    shap_df = pd.read_csv(shap_path) if shap_path.exists() else None
    return explain_visualization_with_openai(api_key, graph_stored_melted_df, shap_df=shap_df, viz_type="shap")

def explain_graph_spatial(api_key):
    global graph_stored_melted_df
    return explain_visualization_with_openai(api_key, graph_stored_melted_df, viz_type="spatial")

def update_heatmap_mm(tissues, cell_types, pathways, use_scaled, color_palette):
    global mm_stored_melted_df, mm_stored_heatmap_fig
    if mm_stored_melted_df is None or mm_stored_melted_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Run pipeline or Load results first", showarrow=False)
        mm_stored_heatmap_fig = fig
        return fig
    df = mm_stored_melted_df.copy()
    if tissues and len(tissues) > 0 and 'All' not in tissues:
        df = df[df['tissue'].isin(tissues)]
    if cell_types and len(cell_types) > 0 and 'All' not in cell_types:
        df = df[df['cell_type'].isin(cell_types)]
    if pathways and len(pathways) > 0 and 'All' not in pathways:
        df = df[df['metabolic_task'].isin(pathways)]
    fig = create_heatmap(df, use_scaled, color_palette)
    mm_stored_heatmap_fig = fig
    return fig

def update_spatial_mm(tissues, celltypes, pathway_sel, color_palette, point_size):
    global mm_stored_spatial_df, mm_stored_spatial_fig
    if mm_stored_spatial_df is None or mm_stored_spatial_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Run pipeline or load pre-computed results first", showarrow=False)
        mm_stored_spatial_fig = fig
        stats_fig = go.Figure()
        stats_fig.add_annotation(text="No data", showarrow=False)
        return fig, [], stats_fig

    fig = create_spatial_plots_from_csv(mm_stored_spatial_df, tissues, celltypes, pathway_sel, color_palette, point_size)
    mm_stored_spatial_fig = fig

    stats_fig = create_spatial_statistics_plots(mm_stored_spatial_df, tissues, celltypes, pathway_sel, color_palette)

    # Get tissue images
    tissue_names = None
    if tissues and 'All' not in tissues:
        tissue_names = [t.split(' (')[0] for t in tissues]
    images = create_tissue_image_gallery(tissue_names if tissue_names else ['All'])

    return fig, images, stats_fig

def update_heatmap_graph(tissues, cell_types, pathways, use_scaled, color_palette):
    global graph_stored_melted_df, graph_stored_heatmap_fig
    if graph_stored_melted_df is None or graph_stored_melted_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Run graph pipeline first", showarrow=False)
        graph_stored_heatmap_fig = fig
        return fig
    df = graph_stored_melted_df.copy()
    if tissues and len(tissues) > 0 and 'All' not in tissues:
        df = df[df['tissue'].isin(tissues)]
    if cell_types and len(cell_types) > 0 and 'All' not in cell_types:
        df = df[df['cell_type'].isin(cell_types)]
    if pathways and len(pathways) > 0 and 'All' not in pathways:
        df = df[df['metabolic_task'].isin(pathways)]
    fig = create_heatmap(df, use_scaled, color_palette)
    graph_stored_heatmap_fig = fig
    return fig

def update_spatial_graph(tissues, celltypes, pathway_sel, color_palette, point_size):
    global graph_stored_spatial_df, graph_stored_spatial_fig
    if graph_stored_spatial_df is None or graph_stored_spatial_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Run pipeline or load pre-computed results first", showarrow=False)
        graph_stored_spatial_fig = fig
        stats_fig = go.Figure()
        stats_fig.add_annotation(text="No data", showarrow=False)
        return fig, [], stats_fig

    fig = create_niche_visualization_from_csv(graph_stored_spatial_df, tissues, celltypes, pathway_sel, color_palette, point_size)
    graph_stored_spatial_fig = fig

    stats_fig = create_niche_statistics_plots(graph_stored_spatial_df, tissues, celltypes, pathway_sel, color_palette)

    # Get tissue images
    tissue_names = None
    if tissues and 'All' not in tissues:
        tissue_names = [t.split(' (')[0] for t in tissues]
    images = create_tissue_image_gallery(tissue_names if tissue_names else ['All'])

    return fig, images, stats_fig

# ============ DEFAULT SAMPLES LIST ============
DEFAULT_SAMPLES = [
    {"name": "1142243F", "count": "1142243F_filtered_count_matrix.tar.gz", "meta": "1142243F_metadata.tar.gz", "spatial": "1142243F_spatial.tar.gz"},
    {"name": "1160920F", "count": "1160920F_filtered_count_matrix.tar.gz", "meta": "1160920F_metadata.tar.gz", "spatial": "1160920F_spatial.tar.gz"},
    {"name": "CID4290", "count": "CID4290_filtered_count_matrix.tar.gz", "meta": "CID4290_metadata.tar.gz", "spatial": "CID4290_spatial.tar.gz"},
    {"name": "CID4465", "count": "CID4465_filtered_count_matrix.tar.gz", "meta": "CID4465_metadata.tar.gz", "spatial": "CID4465_spatial.tar.gz"},
    {"name": "CID4535", "count": "CID4535_filtered_count_matrix.tar.gz", "meta": "CID4535_metadata.tar.gz", "spatial": "CID4535_spatial.tar.gz"},
    {"name": "CID44971", "count": "CID44971_filtered_count_matrix.tar.gz", "meta": "CID44971_metadata.tar.gz", "spatial": "CID44971_spatial.tar.gz"},
]

def process_multiple_samples(sample_configs, batch_size, subset, compute_shap, logger, is_graph=False, k_neighbors=6, n_niches=5):
    """Process multiple samples and combine results."""
    all_adatas = []
    all_per_spot_dfs = []
    all_preds = []
    all_shap_dfs = []

    temp_dir = Path(tempfile.mkdtemp())

    for sample_config in sample_configs:
        sample_name = sample_config['name']
        count_path = sample_config['count']
        spatial_path = sample_config['spatial']
        meta_path = sample_config.get('meta')

        logger.log(f"\n{'='*50}")
        logger.log(f"Processing sample: {sample_name}")
        logger.log(f"{'='*50}")

        try:
            adata, img_path = load_single_sample(
                count_path, spatial_path, meta_path,
                sample_name, temp_dir, logger
            )

            if subset and int(subset) > 0 and int(subset) < adata.n_obs:
                adata = adata[:int(subset)].copy()
                logger.log(f"Subsetted to {adata.n_obs} spots")

            sc.pp.calculate_qc_metrics(adata, percent_top=None, inplace=True)
            sc.pp.filter_genes(adata, min_cells=10)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            adata.raw = adata.copy()
            sc.pp.scale(adata, max_value=10)

            adata = annotate_cell_types(adata, logger)
            gene_tensor = build_gene_tensor(adata, metabolic_genes, logger)

            if img_path and Path(img_path).exists():
                img = image_transform(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)
            else:
                logger.log("No image - using zero embedding")
                img = torch.zeros(1, 3, 224, 224).to(device)

            with torch.no_grad():
                if AMP:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        img_emb_1 = model.encode_image_to_emb(img)
                else:
                    img_emb_1 = model.encode_image_to_emb(img)

            n = adata.n_obs
            preds = np.empty((n, len(PATHWAYS)), dtype=np.float32)
            embeddings_list = []
            bs = int(batch_size)
            if not torch.cuda.is_available():
                bs = min(bs, 16)

            with torch.no_grad():
                for s in range(0, n, bs):
                    e = min(s + bs, n)
                    G = gene_tensor[s:e]
                    if AMP:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            gene_emb = model.encode_genes(G)
                            P = model.fuse_and_predict(gene_emb, img_emb_1)
                            img_exp = img_emb_1.expand(G.shape[0], -1)
                            emb = torch.cat([gene_emb, img_exp], dim=1)
                    else:
                        gene_emb = model.encode_genes(G)
                        P = model.fuse_and_predict(gene_emb, img_emb_1)
                        img_exp = img_emb_1.expand(G.shape[0], -1)
                        emb = torch.cat([gene_emb, img_exp], dim=1)
                    preds[s:e] = P.float().cpu().numpy()
                    embeddings_list.append(emb.float().cpu().numpy())

            embeddings = np.vstack(embeddings_list)

            if is_graph and GRAPH_AVAILABLE and graph_model is not None:
                logger.log("Building spatial graph...")
                coords = adata.obs[['pxl_col_in_fullres', 'pxl_row_in_fullres']].values
                nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='ball_tree').fit(coords)
                distances, indices = nbrs.kneighbors(coords)
                edge_list = []
                for i in range(len(coords)):
                    for j in indices[i][1:]:
                        edge_list.append([i, j])
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().to(device)
                x = torch.tensor(embeddings, dtype=torch.float).to(device)

                logger.log("Running Graph Transformer...")
                with torch.no_grad():
                    graph_embeddings = graph_model(x, edge_index).cpu().numpy()

                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=n_niches, random_state=42)
                niche_labels = kmeans.fit_predict(graph_embeddings)

                # Name niches by dominant cell type
                niche_names = {}
                for niche_id in range(n_niches):
                    niche_mask = niche_labels == niche_id
                    if niche_mask.sum() > 0:
                        niche_celltypes = adata.obs.iloc[np.where(niche_mask)[0]]['cell_type']
                        dominant_ct = niche_celltypes.value_counts().idxmax()
                        niche_names[niche_id] = dominant_ct
                    else:
                        niche_names[niche_id] = "Unknown"

                adata.obs['niche'] = [niche_names[i] for i in niche_labels]
            per_spot_df = make_melted(preds, adata, sample_name, logger)

            if compute_shap:
                gene_tensor_shap = build_gene_tensor(adata, metabolic_genes, logger)
                max_spots = min(50, n)
                shap_df = compute_shap_values(
                    model, gene_tensor_shap, img_emb_1,
                    adata, metabolic_genes, logger,
                    max_spots=max_spots
                )
                if not shap_df.empty:
                    all_shap_dfs.append(shap_df)
                del gene_tensor_shap

            all_adatas.append(adata)
            all_per_spot_dfs.append(per_spot_df)
            all_preds.append(preds)

            logger.log(f"✓ Sample {sample_name} complete: {n} spots")

            del img, gene_tensor, img_emb_1, preds, embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Failed to process {sample_name}: {e}", exc=e)
            continue

    if len(all_adatas) == 0:
        raise ValueError("No samples were successfully processed")

    logger.log("\nCombining results from all samples...")

    combined_adata = ad.concat(all_adatas, join='outer', label='sample', keys=[a.obs['sample'].iloc[0] for a in all_adatas])
    combined_adata.obs['sample'] = combined_adata.obs['sample'].astype(str)

    combined_per_spot = pd.concat(all_per_spot_dfs, ignore_index=True)
    aggregated_df = aggregate_predictions(combined_per_spot, logger)
    combined_shap = pd.concat(all_shap_dfs, ignore_index=True) if all_shap_dfs else pd.DataFrame()

    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except:
        pass

    return combined_adata, combined_per_spot, aggregated_df, combined_shap, all_preds

def run_multimodal_pipeline(count_file, meta_file, spatial_file, batch_size, subset, compute_shap):
    global mm_stored_melted_df, mm_stored_spatial_df
    global mm_stored_heatmap_fig, mm_stored_shap_fig, mm_stored_spatial_fig

    logger = Logger()
    total_t0 = time.time()

    try:
        logger.log("MULTIMODAL PIPELINE STARTING")

        count_path = _normalize_file_arg(count_file)
        spatial_path = _normalize_file_arg(spatial_file)
        meta_path = _normalize_file_arg(meta_file)

        if count_path and spatial_path:
            sample_name = Path(count_path).stem
            for suffix in ['_filtered_count_matrix', '_count_matrix', '_filtered_feature_bc_matrix']:
                sample_name = sample_name.replace(suffix, '')
            sample_name = re.sub(r'[^a-zA-Z0-9_-]', '_', sample_name)

            sample_configs = [{
                'name': sample_name,
                'count': count_path,
                'spatial': spatial_path,
                'meta': meta_path
            }]
        else:
            sample_configs = []
            for sample in DEFAULT_SAMPLES:
                sample_configs.append({
                    'name': sample['name'],
                    'count': str(DEFAULT_FILES_DIR / sample['count']),
                    'spatial': str(DEFAULT_FILES_DIR / sample['spatial']),
                    'meta': str(DEFAULT_FILES_DIR / sample['meta']) if (DEFAULT_FILES_DIR / sample['meta']).exists() else None
                })

        combined_adata, combined_per_spot, aggregated_df, combined_shap, all_preds = process_multiple_samples(
            sample_configs, batch_size, subset, compute_shap, logger, is_graph=False
        )

        # Save files
        melted_path = OUT_DIR / "mm_metabolic_predictions.csv"
        minmax_path = OUT_DIR / "mm_minmax.csv"
        shap_path = OUT_DIR / "mm_shap.csv"
        spatial_csv_path = OUT_DIR / "mm_spatial_data.csv"

        aggregated_df.to_csv(melted_path, index=False)
        make_minmax_transposed_from_preds(all_preds, minmax_path, logger)

        if not combined_shap.empty:
            combined_shap.to_csv(shap_path, index=False)

        spatial_df = make_spatial_csv(combined_adata, combined_per_spot, spatial_csv_path, logger)

        # Store globally
        mm_stored_melted_df = aggregated_df
        mm_stored_spatial_df = spatial_df

        total_time = time.time() - total_t0
        logger.log(f"COMPLETE in {total_time:.2f}s")

        heatmap_fig = create_heatmap(aggregated_df, use_scaled=False, color_palette='RdBu_r')

        if compute_shap and not combined_shap.empty:
            shap_fig = create_shap_plot(shap_path)
        else:
            shap_fig = go.Figure()
            shap_fig.add_annotation(text="SHAP not computed", showarrow=False)

        spatial_fig = create_spatial_plots_from_csv(spatial_df, ['All'], ['All'], 'Glycolysis', 'jet', 6)
        stats_fig = create_spatial_statistics_plots(spatial_df, ['All'], ['All'], 'Glycolysis', 'jet')

        mm_stored_heatmap_fig = heatmap_fig
        mm_stored_shap_fig = shap_fig
        mm_stored_spatial_fig = spatial_fig

        heatmap_insights = generate_heatmap_insights(aggregated_df)
        shap_insights = generate_shap_insights(combined_shap) if not combined_shap.empty else "Enable SHAP checkbox"

        tissues = get_tissue_celltype_labels(spatial_df)
        cell_types_list = ['All'] + sorted(aggregated_df['cell_type'].unique().tolist())
        pathways_list = ['All'] + sorted(aggregated_df['metabolic_task'].unique().tolist())

        # Get initial tissue images
        tissue_images = create_tissue_image_gallery(['All'])

        metrics = f"""COMPLETE

Samples: {', '.join([s['name'] for s in sample_configs])}
Time: {total_time:.1f}s
Device: {device}

Total Spots: {combined_adata.n_obs:,}
Tissues: {aggregated_df['tissue'].nunique()}
Cell Types: {aggregated_df['cell_type'].nunique()}
Pathways: {len(PATHWAYS)}
SHAP: {'Yes' if compute_shap else 'No'}
"""

        shap_file = str(shap_path) if compute_shap and shap_path.exists() else None

        return (
            str(melted_path), str(minmax_path), shap_file, str(spatial_csv_path),
            metrics, heatmap_fig, shap_fig, spatial_fig, tissue_images,
            heatmap_insights, shap_insights,
            gr.update(choices=tissues, value=['All']),
            gr.update(choices=cell_types_list, value=['All']),
            gr.update(choices=pathways_list, value=['All']),
            gr.update(value=False),
            gr.update(choices=PATHWAYS, value='Glycolysis'),
            gr.update(choices=tissues, value=['All']),
            gr.update(choices=cell_types_list, value=['All']),
            logger.get_all()
        )

    except Exception as e:
        logger.error(f"Failed: {e}", exc=e)
        empty_fig = go.Figure()
        empty_fig.add_annotation(text=f"Error: {e}", showarrow=False)
        return (
            None, None, None, None, f"ERROR: {e}", empty_fig, empty_fig, empty_fig, [],
            "", "",
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(),
            logger.get_all()
        )

def run_graph_pipeline(count_file, meta_file, spatial_file, batch_size, subset, k_neighbors, n_niches, compute_shap):
    global graph_stored_melted_df, graph_stored_spatial_df
    global graph_stored_heatmap_fig, graph_stored_shap_fig, graph_stored_spatial_fig

    logger = Logger()
    total_t0 = time.time()

    try:
        logger.log("GRAPH TRANSFORMER PIPELINE STARTING")

        if not GRAPH_AVAILABLE or graph_model is None:
            empty_fig = go.Figure()
            empty_fig.add_annotation(text="Graph Transformer not available", showarrow=False)
            return (
                None, None, None, None,
                "ERROR: Graph Transformer not available",
                empty_fig, empty_fig, empty_fig, [],
                "", "",
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(),
                logger.get_all()
            )

        count_path = _normalize_file_arg(count_file)
        spatial_path = _normalize_file_arg(spatial_file)
        meta_path = _normalize_file_arg(meta_file)

        if count_path and spatial_path:
            sample_name = Path(count_path).stem
            for suffix in ['_filtered_count_matrix', '_count_matrix', '_filtered_feature_bc_matrix']:
                sample_name = sample_name.replace(suffix, '')
            sample_name = re.sub(r'[^a-zA-Z0-9_-]', '_', sample_name)

            sample_configs = [{
                'name': sample_name,
                'count': count_path,
                'spatial': spatial_path,
                'meta': meta_path
            }]
        else:
            sample_configs = []
            for sample in DEFAULT_SAMPLES:
                sample_configs.append({
                    'name': sample['name'],
                    'count': str(DEFAULT_FILES_DIR / sample['count']),
                    'spatial': str(DEFAULT_FILES_DIR / sample['spatial']),
                    'meta': str(DEFAULT_FILES_DIR / sample['meta']) if (DEFAULT_FILES_DIR / sample['meta']).exists() else None
                })

        combined_adata, combined_per_spot, aggregated_df, combined_shap, all_preds = process_multiple_samples(
            sample_configs, batch_size, subset, compute_shap, logger,
            is_graph=True, k_neighbors=k_neighbors, n_niches=n_niches
        )

        melted_path = OUT_DIR / "graph_predictions.csv"
        minmax_path = OUT_DIR / "graph_minmax.csv"
        shap_path = OUT_DIR / "graph_shap.csv"
        spatial_csv_path = OUT_DIR / "graph_spatial_data.csv"

        aggregated_df.to_csv(melted_path, index=False)
        make_minmax_transposed_from_preds(all_preds, minmax_path, logger)

        if not combined_shap.empty:
            combined_shap.to_csv(shap_path, index=False)

        spatial_df = make_spatial_csv(combined_adata, combined_per_spot, spatial_csv_path, logger)

        graph_stored_melted_df = aggregated_df
        graph_stored_spatial_df = spatial_df

        total_time = time.time() - total_t0
        logger.log(f"COMPLETE in {total_time:.2f}s")

        heatmap_fig = create_heatmap(aggregated_df, use_scaled=False, color_palette='RdBu_r')

        if compute_shap and not combined_shap.empty:
            shap_fig = create_shap_plot(shap_path)
        else:
            shap_fig = go.Figure()
            shap_fig.add_annotation(text="SHAP not computed", showarrow=False)

        spatial_fig = create_niche_visualization_from_csv(spatial_df, ['All'], ['All'], 'Glycolysis', 'jet', 6)
        stats_fig = create_niche_statistics_plots(spatial_df, ['All'], ['All'], 'Glycolysis', 'jet')

        graph_stored_heatmap_fig = heatmap_fig
        graph_stored_shap_fig = shap_fig
        graph_stored_spatial_fig = spatial_fig

        heatmap_insights = generate_heatmap_insights(aggregated_df)
        shap_insights = generate_shap_insights(combined_shap) if not combined_shap.empty else "Enable SHAP checkbox"

        tissues = get_tissue_celltype_labels(spatial_df)
        cell_types_list = ['All'] + sorted(aggregated_df['cell_type'].unique().tolist())
        pathways_list = ['All'] + sorted(aggregated_df['metabolic_task'].unique().tolist())

        tissue_images = create_tissue_image_gallery(['All'])

        metrics = f"""COMPLETE

Samples: {', '.join([s['name'] for s in sample_configs])}
Time: {total_time:.1f}s
Device: {device}

Total Spots: {combined_adata.n_obs:,}
Tissues: {aggregated_df['tissue'].nunique()}
Cell Types: {aggregated_df['cell_type'].nunique()}
Niches: {n_niches}
Graph: k={k_neighbors} neighbors
SHAP: {'Yes' if compute_shap else 'No'}
"""

        shap_file = str(shap_path) if compute_shap and shap_path.exists() else None

        return (
            str(melted_path), str(minmax_path), shap_file, str(spatial_csv_path),
            metrics, heatmap_fig, shap_fig, spatial_fig, tissue_images, stats_fig,
            heatmap_insights, shap_insights,
            gr.update(choices=tissues, value=['All']),
            gr.update(choices=cell_types_list, value=['All']),
            gr.update(choices=pathways_list, value=['All']),
            gr.update(value=False),
            gr.update(choices=PATHWAYS, value='Glycolysis'),
            gr.update(choices=tissues, value=['All']),
            gr.update(choices=cell_types_list, value=['All']),
            logger.get_all()
        )

    except Exception as e:
        logger.error(f"Failed: {e}", exc=e)
        empty_fig = go.Figure()
        empty_fig.add_annotation(text=f"Error: {e}", showarrow=False)
        return (
            None, None, None, None, f"ERROR: {e}", empty_fig, empty_fig, empty_fig, [], empty_fig,
            "", "",
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(),
            logger.get_all()
        )

def load_precomputed_results_mm():
    global mm_stored_melted_df, mm_stored_spatial_df
    global mm_stored_heatmap_fig, mm_stored_shap_fig, mm_stored_spatial_fig

    logger = Logger()
    logger.log("Loading pre-computed multimodal results...")

    melted_path = RESULTS_DIR / "mm_metabolic_predictions.csv"
    minmax_path = RESULTS_DIR / "mm_minmax.csv"
    shap_path = RESULTS_DIR / "mm_shap.csv"
    spatial_path = RESULTS_DIR / "mm_spatial_data.csv"

    if not melted_path.exists():
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="Pre-computed results not found", showarrow=False)
        return (
            None, None, None, None,
            "ERROR: Pre-computed CSV files not found.",
            empty_fig, empty_fig, empty_fig, [],
            "", "",
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
            logger.get_all()
        )

    melted_df = pd.read_csv(melted_path)

    if 'trimean' not in melted_df.columns:
        melted_df = aggregate_predictions(melted_df, logger)

    mm_stored_melted_df = melted_df

    # Load spatial CSV
    if spatial_path.exists():
        spatial_df = pd.read_csv(spatial_path)
        mm_stored_spatial_df = spatial_df
    else:
        spatial_df = None
        mm_stored_spatial_df = None

    heatmap_fig = create_heatmap(melted_df, use_scaled=False, color_palette='RdBu_r')
    mm_stored_heatmap_fig = heatmap_fig

    if shap_path.exists():
        shap_fig = create_shap_plot(shap_path)
        shap_df = pd.read_csv(shap_path)
        shap_insights = generate_shap_insights(shap_df)
        shap_file = str(shap_path)
    else:
        shap_fig = go.Figure()
        shap_fig.add_annotation(text="No SHAP CSV found", showarrow=False)
        shap_insights = "No SHAP data available."
        shap_file = None
    mm_stored_shap_fig = shap_fig

    if spatial_df is not None:
        spatial_fig = create_spatial_plots_from_csv(spatial_df, ['All'], ['All'], 'Glycolysis', 'jet', 6)
        stats_fig = create_spatial_statistics_plots(spatial_df, ['All'], ['All'], 'Glycolysis', 'jet')
    else:
        spatial_fig = go.Figure()
        spatial_fig.add_annotation(text="No spatial CSV found", showarrow=False)
        stats_fig = go.Figure()
        stats_fig.add_annotation(text="No spatial data", showarrow=False)
    mm_stored_spatial_fig = spatial_fig

    heatmap_insights = generate_heatmap_insights(melted_df)

    tissues = get_tissue_celltype_labels(spatial_df) if spatial_df is not None else ['All']
    cell_types_list = ['All'] + sorted(melted_df['cell_type'].unique().tolist())
    pathways_list = ['All'] + sorted(melted_df['metabolic_task'].unique().tolist())

    spatial_file = str(spatial_path) if spatial_path.exists() else None
    tissue_images = create_tissue_image_gallery(['All'])

    metrics = f"""LOADED PRE-COMPUTED RESULTS

Tissues: {melted_df['tissue'].nunique()}
Cell Types: {melted_df['cell_type'].nunique()}
Pathways: {melted_df['metabolic_task'].nunique()}
"""

    logger.log("Pre-computed results loaded successfully.")
    return (
        str(melted_path), str(minmax_path) if minmax_path.exists() else None, shap_file, spatial_file,
        metrics, heatmap_fig, shap_fig, spatial_fig, tissue_images, stats_fig,
        heatmap_insights, shap_insights,
        gr.update(choices=tissues, value=['All']),
        gr.update(choices=cell_types_list, value=['All']),
        gr.update(choices=pathways_list, value=['All']),
        gr.update(value=False),
        gr.update(choices=PATHWAYS, value='Glycolysis'),
        gr.update(choices=tissues, value=['All']),
        gr.update(choices=cell_types_list, value=['All']),
        logger.get_all()
    )

def load_precomputed_results_graph():
    global graph_stored_melted_df, graph_stored_spatial_df
    global graph_stored_heatmap_fig, graph_stored_shap_fig, graph_stored_spatial_fig

    logger = Logger()
    logger.log("Loading pre-computed graph results...")

    melted_path = RESULTS_DIR / "graph_predictions.csv"
    minmax_path = RESULTS_DIR / "graph_minmax.csv"
    shap_path = RESULTS_DIR / "graph_shap.csv"
    spatial_path = RESULTS_DIR / "graph_spatial_data.csv"

    if not melted_path.exists():
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="Pre-computed results not found", showarrow=False)
        return (
            None, None, None, None,
            "ERROR: Pre-computed CSV files not found.",
            empty_fig, empty_fig, empty_fig, [],
            "", "",
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
            logger.get_all()
        )

    melted_df = pd.read_csv(melted_path)

    if 'trimean' not in melted_df.columns:
        melted_df = aggregate_predictions(melted_df, logger)

    graph_stored_melted_df = melted_df

    if spatial_path.exists():
        spatial_df = pd.read_csv(spatial_path)
        graph_stored_spatial_df = spatial_df
    else:
        spatial_df = None
        graph_stored_spatial_df = None

    heatmap_fig = create_heatmap(melted_df, use_scaled=False, color_palette='RdBu_r')
    graph_stored_heatmap_fig = heatmap_fig

    if shap_path.exists():
        shap_fig = create_shap_plot(shap_path)
        shap_df = pd.read_csv(shap_path)
        shap_insights = generate_shap_insights(shap_df)
        shap_file = str(shap_path)
    else:
        shap_fig = go.Figure()
        shap_fig.add_annotation(text="No SHAP CSV found", showarrow=False)
        shap_insights = "No SHAP data available."
        shap_file = None
    graph_stored_shap_fig = shap_fig

    if spatial_df is not None:
        spatial_fig = create_niche_visualization_from_csv(spatial_df, ['All'], ['All'], 'Glycolysis', 'jet', 6)
        stats_fig = create_niche_statistics_plots(spatial_df, ['All'], ['All'], 'Glycolysis', 'jet')
    else:
        spatial_fig = go.Figure()
        spatial_fig.add_annotation(text="No spatial CSV found", showarrow=False)
        stats_fig = go.Figure()
        stats_fig.add_annotation(text="No spatial data", showarrow=False)
    graph_stored_spatial_fig = spatial_fig

    heatmap_insights = generate_heatmap_insights(melted_df)

    tissues = get_tissue_celltype_labels(spatial_df) if spatial_df is not None else ['All']
    cell_types_list = ['All'] + sorted(melted_df['cell_type'].unique().tolist())
    pathways_list = ['All'] + sorted(melted_df['metabolic_task'].unique().tolist())

    spatial_file = str(spatial_path) if spatial_path.exists() else None
    tissue_images = create_tissue_image_gallery(['All'])

    metrics = f"""LOADED PRE-COMPUTED RESULTS

Tissues: {melted_df['tissue'].nunique()}
Cell Types: {melted_df['cell_type'].nunique()}
Pathways: {melted_df['metabolic_task'].nunique()}
"""

    logger.log("Pre-computed results loaded successfully.")
    return (
        str(melted_path), str(minmax_path) if minmax_path.exists() else None, shap_file, spatial_file,
        metrics, heatmap_fig, shap_fig, spatial_fig, tissue_images, stats_fig,
        heatmap_insights, shap_insights,
        gr.update(choices=tissues, value=['All']),
        gr.update(choices=cell_types_list, value=['All']),
        gr.update(choices=pathways_list, value=['All']),
        gr.update(value=False),
        gr.update(choices=PATHWAYS, value='Glycolysis'),
        gr.update(choices=tissues, value=['All']),
        gr.update(choices=cell_types_list, value=['All']),
        logger.get_all()
    )

def load_default_and_run_mm():
    """Load default samples and run multimodal pipeline."""
    return run_multimodal_pipeline(None, None, None, 32, 0, True)

def load_default_and_run_graph():
    """Load default samples and run graph pipeline."""
    return run_graph_pipeline(None, None, None, 32, 0, 6, 5, True)

# ============ GRADIO UI ============

css = """
.plot-container {
    width: 100%;
    height: 750px;
    overflow: auto !important;
    border: 2px solid #e0e0e0;
    border-radius: 8px;
    background: white;
}
.description-box {
    padding: 20px;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-left: 4px solid #4CAF50;
    margin-bottom: 20px;
    border-radius: 8px;
    color: #e0e0e0 !important;
}
.description-box h3 {
    color: #4CAF50 !important;
    margin-top: 0;
}
.description-box p, .description-box li, .description-box code {
    color: #e0e0e0 !important;
}
.description-box code {
    background: #2d2d44;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: monospace;
}
.description-box strong {
    color: #81c784 !important;
}
.folder-structure {
    background: #2d2d44;
    padding: 15px;
    border-radius: 6px;
    font-family: monospace;
    font-size: 12px;
    margin: 10px 0;
    color: #b0b0b0 !important;
}
.tissue-gallery {
    display: flex;
    flex-wrap: nowrap;
    overflow-x: auto;
    gap: 10px;
    padding: 10px 0;
}
.description-box h4 {
    color: #4CAF50 !important;
    margin: 0 0 8px 0;
    font-size: 14px;
}
.description-box p {
    margin: 0;
    line-height: 1.4;
}
"""

with gr.Blocks(title="MetaboNiche", theme=gr.themes.Soft(), css=css) as demo:
    gr.Markdown("# 🧬 MetaboNiche: Spatial Metabolic Niche Discovery")

    gr.HTML("""
    <div class="description-box">
        <h3>🔬 Research Objective</h3>
        <p>A multimodal Graph-Transformer model integrating <strong>histology images</strong>, <strong>spatial coordinates</strong>,
        and <strong>gene expression</strong> to identify spatially coherent metabolic niches in tissue.
        Predicts <strong>54 metabolic pathway activities</strong> for tumor microenvironment analysis.</p>
    </div>
    """)

    with gr.Tabs():
        with gr.Tab("🔬 Multimodal Model"):
            with gr.Tabs():
                with gr.Tab("Pipeline"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.HTML("""
                            <div class="description-box" style="height: 100%;">
                                <h4>📁 Data Format</h4>
                                <p style="font-size: 12px; font-family: monospace;">
                                <strong>Count:</strong> matrix.mtx.gz, features.tsv.gz, barcodes.tsv.gz<br>
                                <strong>Spatial:</strong> tissue_positions_list.csv, tissue_hires_image.png
                                </p>
                            </div>
                            """)
                        with gr.Column(scale=1):
                            gr.HTML("""
                            <div class="description-box" style="height: 100%;">
                                <h4>⚡ Quick Start</h4>
                                <p style="font-size: 12px;">
                                • <strong>Load Default:</strong> Process 6 pre-configured samples<br>
                                • <strong>Pre-computed:</strong> View saved results instantly<br>
                                • <strong>Upload:</strong> Your own .tar.gz/.zip files
                                </p>
                            </div>
                            """)
                        with gr.Column(scale=1):
                            gr.HTML("""
                            <div class="description-box" style="height: 100%;">
                                <h4>📊 Outputs</h4>
                                <p style="font-size: 12px;">
                                • Heatmap: Cell type × Pathway activity<br>
                                • SHAP: Gene importance scores<br>
                                • Spatial: Coordinate visualizations<br>
                                • CSVs: Downloadable results
                                </p>
                            </div>
                            """)
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 📂 Data Input")
                            mm_load_default_btn = gr.Button("📂 Load Default (Multi-Sample)", variant="primary", size="lg")
                            mm_load_results_btn = gr.Button("📊 Load Pre-computed", variant="secondary", size="sm")

                            gr.Markdown("---")
                            gr.Markdown("### 📤 Upload Custom Sample")
                            mm_count = gr.File(label="Count Matrix (.tar.gz or .zip)", file_types=[".tar.gz", ".gz", ".zip"])
                            mm_count_status = gr.Markdown("Waiting...")
                            mm_meta = gr.File(label="Metadata (.tar.gz or .zip) [Optional]", file_types=[".tar.gz", ".gz", ".zip"])
                            mm_meta_status = gr.Markdown("Waiting...")
                            mm_spatial = gr.File(label="Spatial (.tar.gz or .zip)", file_types=[".tar.gz", ".gz", ".zip"])
                            mm_spatial_status = gr.Markdown("Waiting...")
                            mm_count.change(on_file_change, inputs=mm_count, outputs=mm_count_status)
                            mm_meta.change(on_file_change, inputs=mm_meta, outputs=mm_meta_status)
                            mm_spatial.change(on_file_change, inputs=mm_spatial, outputs=mm_spatial_status)

                            gr.Markdown("### ⚙️ Settings")
                            mm_batch = gr.Slider(4, 128, value=32, step=4, label="Batch size")
                            mm_subset = gr.Number(label="Subset spots (0 = all)", value=0, precision=0)
                            mm_shap = gr.Checkbox(label="Compute SHAP (gene importance)", value=True)
                            mm_btn = gr.Button("▶️ Run Pipeline (Uploaded)", variant="secondary", size="lg")
                        with gr.Column(scale=2):
                            gr.Markdown("### 📋 Results Summary")
                            mm_metrics = gr.Textbox(label="Pipeline Status", lines=16)
                    with gr.Row():
                        mm_melted_dl = gr.File(label="📄 Predictions CSV")
                        mm_minmax_dl = gr.File(label="📄 MinMax CSV")
                        mm_shap_dl = gr.File(label="📄 SHAP CSV")
                        mm_spatial_dl = gr.File(label="📄 Spatial CSV")
                    with gr.Accordion("📜 Processing Log", open=False):
                        mm_log = gr.Textbox(lines=20)

                with gr.Tab("📊 Heatmap"):
                    with gr.Row():
                        mm_tissue_filter = gr.Dropdown(choices=['All'], value=['All'], multiselect=True, label="Filter by Tissue")
                        mm_celltype_filter = gr.Dropdown(choices=['All'], value=['All'], multiselect=True, label="Filter by Cell Type")
                        mm_pathway_filter = gr.Dropdown(choices=['All'], value=['All'], multiselect=True, label="Filter by Pathway")
                    with gr.Row():
                        mm_scaled = gr.Checkbox(label="Use Z-Score Scaling", value=False)
                        mm_palette = gr.Dropdown(choices=list(COLOR_PALETTES.keys()), value='RdBu_r', label="Color Palette")
                    mm_heatmap = gr.Plot(label="Metabolic Heatmap", elem_classes="plot-container")
                    gr.Markdown("### 🔍 Automated Insights")
                    mm_heatmap_insights = gr.Markdown("Run pipeline first...")
                    with gr.Accordion("🤖 AI-Powered Explanation (GPT-4)", open=False):
                        with gr.Row():
                            mm_heatmap_openai_key = gr.Textbox(label="OpenAI API Key", placeholder="sk-...", type="password")
                            mm_heatmap_explain_btn = gr.Button("🔍 Generate Explanation", variant="primary")
                        mm_heatmap_explanation = gr.Markdown("")

                with gr.Tab("🧬 SHAP Analysis"):
                    mm_shap_plot = gr.Plot(label="Gene Importance", elem_classes="plot-container")
                    gr.Markdown("### 🔬 Gene Importance Interpretation")
                    mm_shap_insights = gr.Markdown("Enable SHAP checkbox and run pipeline...")
                    with gr.Accordion("🤖 AI-Powered Explanation (GPT-4)", open=False):
                        with gr.Row():
                            mm_shap_openai_key = gr.Textbox(label="OpenAI API Key", placeholder="sk-...", type="password")
                            mm_shap_explain_btn = gr.Button("🔍 Generate Explanation", variant="primary")
                        mm_shap_explanation = gr.Markdown("")

                with gr.Tab("🗺️ Spatial Visualization"):
                    gr.Markdown("### 🎯 Filters")
                    with gr.Row():
                        mm_spatial_tissue = gr.Dropdown(choices=['All'], value=['All'], multiselect=True, label="Filter by Tissue (Cell Types)")
                        mm_spatial_celltype = gr.Dropdown(choices=['All'], value=['All'], multiselect=True, label="Filter by Cell Type")
                        mm_pathway_selector = gr.Dropdown(choices=PATHWAYS, value='Glycolysis', label="Select Pathway")
                    with gr.Row():
                        mm_spatial_palette = gr.Dropdown(
                            choices=['jet', 'viridis', 'plasma', 'inferno', 'magma', 'RdBu_r', 'Spectral_r', 'coolwarm', 'YlOrRd', 'turbo'],
                            value='jet',
                            label="Color Palette"
                        )
                        mm_spatial_point_size = gr.Slider(3, 15, value=6, step=1, label="Point Size")

                    gr.Markdown("### 📊 Spatial Plots")
                    mm_spatial_plot = gr.Plot(label="Predicted vs Ground Truth", elem_classes="plot-container")

                    gr.Markdown("### 🖼️ Tissue Images")
                    mm_tissue_gallery = gr.Gallery(label="H&E Tissue Images", columns=6, height=250, object_fit="contain", allow_preview=True)

                    gr.Markdown("### 📈 Summary Statistics")
                    mm_stats_plot = gr.Plot(label="Distribution & Composition")

                    with gr.Accordion("🤖 AI-Powered Explanation (GPT-4)", open=False):
                        with gr.Row():
                            mm_spatial_openai_key = gr.Textbox(label="OpenAI API Key", placeholder="sk-...", type="password")
                            mm_spatial_explain_btn = gr.Button("🔍 Generate Explanation", variant="primary")
                        mm_spatial_explanation = gr.Markdown("")

            # Event handlers for MM
            for comp in [mm_tissue_filter, mm_celltype_filter, mm_pathway_filter, mm_scaled, mm_palette]:
                comp.change(
                    fn=update_heatmap_mm,
                    inputs=[mm_tissue_filter, mm_celltype_filter, mm_pathway_filter, mm_scaled, mm_palette],
                    outputs=mm_heatmap
                )

            # Update cell type dropdown when tissue changes
            mm_spatial_tissue.change(
                fn=lambda tissues: update_celltype_dropdown(tissues, mm_stored_spatial_df),
                inputs=[mm_spatial_tissue],
                outputs=[mm_spatial_celltype]
            )

            # Update spatial plot when any filter changes
            for comp in [mm_spatial_tissue, mm_spatial_celltype, mm_pathway_selector, mm_spatial_palette, mm_spatial_point_size]:
                comp.change(
                    fn=update_spatial_mm,
                    inputs=[mm_spatial_tissue, mm_spatial_celltype, mm_pathway_selector, mm_spatial_palette, mm_spatial_point_size],
                    outputs=[mm_spatial_plot, mm_tissue_gallery, mm_stats_plot]
                )

            mm_load_default_btn.click(
                fn=load_default_and_run_mm,
                outputs=[
                    mm_melted_dl, mm_minmax_dl, mm_shap_dl, mm_spatial_dl,
                    mm_metrics, mm_heatmap, mm_shap_plot, mm_spatial_plot, mm_tissue_gallery, mm_stats_plot,
                    mm_heatmap_insights, mm_shap_insights,
                    mm_tissue_filter, mm_celltype_filter, mm_pathway_filter, mm_scaled,
                    mm_pathway_selector, mm_spatial_tissue, mm_spatial_celltype,
                    mm_log
                ]
            )

            mm_load_results_btn.click(
                fn=load_precomputed_results_mm,
                outputs=[
                    mm_melted_dl, mm_minmax_dl, mm_shap_dl, mm_spatial_dl,
                    mm_metrics, mm_heatmap, mm_shap_plot, mm_spatial_plot, mm_tissue_gallery, mm_stats_plot,
                    mm_heatmap_insights, mm_shap_insights,
                    mm_tissue_filter, mm_celltype_filter, mm_pathway_filter, mm_scaled,
                    mm_pathway_selector, mm_spatial_tissue, mm_spatial_celltype,
                    mm_log
                ]
            )

            mm_btn.click(
                fn=run_multimodal_pipeline,
                inputs=[mm_count, mm_meta, mm_spatial, mm_batch, mm_subset, mm_shap],
                outputs=[
                    mm_melted_dl, mm_minmax_dl, mm_shap_dl, mm_spatial_dl,
                    mm_metrics, mm_heatmap, mm_shap_plot, mm_spatial_plot, mm_tissue_gallery, mm_stats_plot,
                    mm_heatmap_insights, mm_shap_insights,
                    mm_tissue_filter, mm_celltype_filter, mm_pathway_filter, mm_scaled,
                    mm_pathway_selector, mm_spatial_tissue, mm_spatial_celltype,
                    mm_log
                ]
            )

            mm_heatmap_explain_btn.click(fn=explain_mm_heatmap, inputs=[mm_heatmap_openai_key], outputs=mm_heatmap_explanation)
            mm_shap_explain_btn.click(fn=explain_mm_shap, inputs=[mm_shap_openai_key], outputs=mm_shap_explanation)
            mm_spatial_explain_btn.click(fn=explain_mm_spatial, inputs=[mm_spatial_openai_key], outputs=mm_spatial_explanation)

        if GRAPH_AVAILABLE and graph_model is not None:
            with gr.Tab("🧠 Graph Transformer"):
                with gr.Tabs():
                    with gr.Tab("Pipeline"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.HTML("""
                                <div class="description-box" style="height: 100%;">
                                    <h4>📁 Data Format</h4>
                                    <p style="font-size: 12px; font-family: monospace;">
                                    <strong>Count:</strong> matrix.mtx.gz, features.tsv.gz, barcodes.tsv.gz<br>
                                    <strong>Spatial:</strong> tissue_positions_list.csv, tissue_hires_image.png
                                    </p>
                                </div>
                                """)
                            with gr.Column(scale=1):
                                gr.HTML("""
                                <div class="description-box" style="height: 100%;">
                                    <h4>⚡ Quick Start</h4>
                                    <p style="font-size: 12px;">
                                    • <strong>Load Default:</strong> Process 6 pre-configured samples<br>
                                    • <strong>Pre-computed:</strong> View saved results instantly<br>
                                    • <strong>Upload:</strong> Your own .tar.gz/.zip files
                                    </p>
                                </div>
                                """)
                            with gr.Column(scale=1):
                                gr.HTML("""
                                <div class="description-box" style="height: 100%;">
                                    <h4>📊 Outputs</h4>
                                    <p style="font-size: 12px;">
                                    • Heatmap: Cell type × Pathway activity<br>
                                    • SHAP: Gene importance scores<br>
                                    • Spatial: Coordinate visualizations<br>
                                    • CSVs: Downloadable results
                                    </p>
                                </div>
                                """)
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### 📂 Data Input")
                                graph_load_default_btn = gr.Button("📂 Load Default (Multi-Sample)", variant="primary", size="lg")
                                graph_load_results_btn = gr.Button("📊 Load Pre-computed", variant="secondary", size="sm")

                                gr.Markdown("---")
                                gr.Markdown("### 📤 Upload Custom Sample")
                                graph_count = gr.File(label="Count Matrix (.tar.gz or .zip)", file_types=[".tar.gz", ".gz", ".zip"])
                                graph_count_status = gr.Markdown("Waiting...")
                                graph_meta = gr.File(label="Metadata [Optional]", file_types=[".tar.gz", ".gz", ".zip"])
                                graph_meta_status = gr.Markdown("Waiting...")
                                graph_spatial = gr.File(label="Spatial (.tar.gz or .zip)", file_types=[".tar.gz", ".gz", ".zip"])
                                graph_spatial_status = gr.Markdown("Waiting...")
                                graph_count.change(on_file_change, inputs=graph_count, outputs=graph_count_status)
                                graph_meta.change(on_file_change, inputs=graph_meta, outputs=graph_meta_status)
                                graph_spatial.change(on_file_change, inputs=graph_spatial, outputs=graph_spatial_status)

                                gr.Markdown("### ⚙️ Settings")
                                graph_batch = gr.Slider(4, 128, value=32, step=4, label="Batch size")
                                graph_subset = gr.Number(label="Subset spots (0 = all)", value=0, precision=0)
                                graph_k = gr.Slider(3, 15, value=6, step=1, label="k-NN neighbors for graph")
                                graph_niches = gr.Slider(2, 10, value=5, step=1, label="Number of niches to discover")
                                graph_shap = gr.Checkbox(label="Compute SHAP (gene importance)", value=True)
                                graph_btn = gr.Button("▶️ Run Pipeline (Uploaded)", variant="secondary", size="lg")
                            with gr.Column(scale=2):
                                gr.Markdown("### 📋 Results Summary")
                                graph_metrics = gr.Textbox(label="Pipeline Status", lines=16)
                        with gr.Row():
                            graph_melted_dl = gr.File(label="📄 Predictions CSV")
                            graph_minmax_dl = gr.File(label="📄 MinMax CSV")
                            graph_shap_dl = gr.File(label="📄 SHAP CSV")
                            graph_spatial_dl = gr.File(label="📄 Spatial CSV")
                        with gr.Accordion("📜 Processing Log", open=False):
                            graph_log = gr.Textbox(lines=20)

                    with gr.Tab("📊 Heatmap"):
                        with gr.Row():
                            graph_tissue_filter = gr.Dropdown(choices=['All'], value=['All'], multiselect=True, label="Filter by Tissue")
                            graph_celltype_filter = gr.Dropdown(choices=['All'], value=['All'], multiselect=True, label="Filter by Cell Type")
                            graph_pathway_filter = gr.Dropdown(choices=['All'], value=['All'], multiselect=True, label="Filter by Pathway")
                        with gr.Row():
                            graph_scaled = gr.Checkbox(label="Use Z-Score Scaling", value=False)
                            graph_palette = gr.Dropdown(choices=list(COLOR_PALETTES.keys()), value='RdBu_r', label="Color Palette")
                        graph_heatmap = gr.Plot(label="Metabolic Heatmap", elem_classes="plot-container")
                        gr.Markdown("### 🔍 Automated Insights")
                        graph_heatmap_insights = gr.Markdown("Run pipeline first...")
                        with gr.Accordion("🤖 AI-Powered Explanation (GPT-4)", open=False):
                            with gr.Row():
                                graph_heatmap_openai_key = gr.Textbox(label="OpenAI API Key", placeholder="sk-...", type="password")
                                graph_heatmap_explain_btn = gr.Button("🔍 Generate Explanation", variant="primary")
                            graph_heatmap_explanation = gr.Markdown("")

                    with gr.Tab("🧬 SHAP Analysis"):
                        graph_shap_plot = gr.Plot(label="Gene Importance", elem_classes="plot-container")
                        gr.Markdown("### 🔬 Gene Importance Interpretation")
                        graph_shap_insights = gr.Markdown("Enable SHAP checkbox and run pipeline...")
                        with gr.Accordion("🤖 AI-Powered Explanation (GPT-4)", open=False):
                            with gr.Row():
                                graph_shap_openai_key = gr.Textbox(label="OpenAI API Key", placeholder="sk-...", type="password")
                                graph_shap_explain_btn = gr.Button("🔍 Generate Explanation", variant="primary")
                            graph_shap_explanation = gr.Markdown("")

                    with gr.Tab("🗺️ Spatial & Niches"):
                        gr.Markdown("### 🎯 Filters")
                        with gr.Row():
                            graph_spatial_tissue = gr.Dropdown(choices=['All'], value=['All'], multiselect=True, label="Filter by Tissue (Cell Types)")
                            graph_spatial_celltype = gr.Dropdown(choices=['All'], value=['All'], multiselect=True, label="Filter by Cell Type/Niche")
                            graph_pathway_selector = gr.Dropdown(choices=PATHWAYS, value='Glycolysis', label="Select Pathway")
                        with gr.Row():
                            graph_spatial_palette = gr.Dropdown(
                                choices=['jet', 'viridis', 'plasma', 'inferno', 'magma', 'RdBu_r', 'Spectral_r', 'coolwarm', 'YlOrRd', 'turbo'],
                                value='jet',
                                label="Color Palette"
                            )
                            graph_spatial_point_size = gr.Slider(3, 15, value=6, step=1, label="Point Size")

                        gr.Markdown("### 📊 Spatial Plots")
                        graph_spatial_plot = gr.Plot(label="Predicted vs Ground Truth", elem_classes="plot-container")

                        gr.Markdown("### 🖼️ Tissue Images")
                        graph_tissue_gallery = gr.Gallery(label="H&E Tissue Images", columns=6, height=250, object_fit="contain", allow_preview=True)

                        gr.Markdown("### 📈 Summary Statistics")
                        graph_stats_plot = gr.Plot(label="Distribution & Composition")

                        with gr.Accordion("🤖 AI-Powered Explanation (GPT-4)", open=False):
                            with gr.Row():
                                graph_spatial_openai_key = gr.Textbox(label="OpenAI API Key", placeholder="sk-...", type="password")
                                graph_spatial_explain_btn = gr.Button("🔍 Generate Explanation", variant="primary")
                            graph_spatial_explanation = gr.Markdown("")


                # Event handlers for Graph
                for comp in [graph_tissue_filter, graph_celltype_filter, graph_pathway_filter, graph_scaled, graph_palette]:
                    comp.change(
                        fn=update_heatmap_graph,
                        inputs=[graph_tissue_filter, graph_celltype_filter, graph_pathway_filter, graph_scaled, graph_palette],
                        outputs=graph_heatmap
                    )

                # Update cell type dropdown when tissue changes
                graph_spatial_tissue.change(
                    fn=lambda tissues: update_celltype_dropdown(tissues, graph_stored_spatial_df),
                    inputs=[graph_spatial_tissue],
                    outputs=[graph_spatial_celltype]
                )

                # Update spatial plot when any filter changes
                for comp in [graph_spatial_tissue, graph_spatial_celltype, graph_pathway_selector, graph_spatial_palette, graph_spatial_point_size]:
                    comp.change(
                        fn=update_spatial_graph,
                        inputs=[graph_spatial_tissue, graph_spatial_celltype, graph_pathway_selector, graph_spatial_palette, graph_spatial_point_size],
                        outputs=[graph_spatial_plot, graph_tissue_gallery, graph_stats_plot]
                    )

                graph_load_default_btn.click(
                    fn=load_default_and_run_graph,
                    outputs=[
                        graph_melted_dl, graph_minmax_dl, graph_shap_dl, graph_spatial_dl,
                        graph_metrics, graph_heatmap, graph_shap_plot, graph_spatial_plot, graph_tissue_gallery, graph_stats_plot,
                        graph_heatmap_insights, graph_shap_insights,
                        graph_tissue_filter, graph_celltype_filter, graph_pathway_filter, graph_scaled,
                        graph_pathway_selector, graph_spatial_tissue, graph_spatial_celltype,
                        graph_log
                    ]
                )

                graph_load_results_btn.click(
                    fn=load_precomputed_results_graph,
                    outputs=[
                        graph_melted_dl, graph_minmax_dl, graph_shap_dl, graph_spatial_dl,
                        graph_metrics, graph_heatmap, graph_shap_plot, graph_spatial_plot, graph_tissue_gallery, graph_stats_plot,
                        graph_heatmap_insights, graph_shap_insights,
                        graph_tissue_filter, graph_celltype_filter, graph_pathway_filter, graph_scaled,
                        graph_pathway_selector, graph_spatial_tissue, graph_spatial_celltype,
                        graph_log
                    ]
                )

                graph_btn.click(
                    fn=run_graph_pipeline,
                    inputs=[graph_count, graph_meta, graph_spatial, graph_batch, graph_subset,
                            graph_k, graph_niches, graph_shap],
                    outputs=[
                        graph_melted_dl, graph_minmax_dl, graph_shap_dl, graph_spatial_dl,
                        graph_metrics, graph_heatmap, graph_shap_plot, graph_spatial_plot, graph_tissue_gallery, graph_stats_plot,
                        graph_heatmap_insights, graph_shap_insights,
                        graph_tissue_filter, graph_celltype_filter, graph_pathway_filter, graph_scaled,
                        graph_pathway_selector, graph_spatial_tissue, graph_spatial_celltype,
                        graph_log
                    ]
                )

                graph_heatmap_explain_btn.click(fn=explain_graph_heatmap, inputs=[graph_heatmap_openai_key], outputs=graph_heatmap_explanation)
                graph_shap_explain_btn.click(fn=explain_graph_shap, inputs=[graph_shap_openai_key], outputs=graph_shap_explanation)
                graph_spatial_explain_btn.click(fn=explain_graph_spatial, inputs=[graph_spatial_openai_key], outputs=graph_spatial_explanation)

demo.queue(max_size=5)
if __name__ == "__main__":
    demo.launch()
