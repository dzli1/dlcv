from pathlib import Path
import torch

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = DATA_DIR / "figures"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "runs"
METADATA_PARQUET = PROCESSED_DIR / "wikiart_metadata.parquet"
SPLIT_METADATA_PARQUET = PROCESSED_DIR / "wikiart_splits.parquet"
LABEL_MAPPING_FILE = PROCESSED_DIR / "label_mappings.json"

# Ensure directories exist at import time for smoother CLI usage
for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DIR, FIGURES_DIR, CHECKPOINT_DIR, LOG_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Hugging Face dataset configuration
HF_DATASET_NAME = "huggan/wikiart"
HF_DATASET_CONFIG = "default"
HF_CACHE_DIR = RAW_DATA_DIR / "hf_cache"
HF_DATASET_SPLIT = "train"

# Dataset filtering + sampling
MIN_YEAR = 1400
MAX_YEAR = 1970
MIN_RESOLUTION = 512  # shortest edge in pixels
MIN_MOVEMENT_SAMPLES = 400
TARGET_NUM_MOVEMENTS = 12
MIN_ARTIST_SAMPLES = 25
TARGET_NUM_ARTISTS = 300
ALLOW_ARTIST_OTHER_BUCKET = True
ARTIST_OTHER_LABEL = "other"
PRIMARY_MEDIUMS = {"oil", "acrylic", "tempera", "mixed", "canvas", "panel"}
PRIMARY_GENRES = {
    "portrait", "religious painting", "landscape", "genre painting",
    "mythological painting", "historical painting", "cityscape",
    "still life", "marina"
}

# Splitting
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 1337
GROUP_SPLIT_FOLDS = 1  # we only need a single disjoint split for now

# Preprocessing
IMAGE_SIZE_CNN = 224
IMAGE_SIZE_VIT = 384
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training defaults
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
VAL_BATCH_SIZE = 64
NUM_WORKERS = 8
PIN_MEMORY = True
GRAD_ACCUMULATION_STEPS = 1
BASE_LR = 2e-4
WEIGHT_DECAY = 0.05
EPOCHS = 25
LR_SCHEDULER = "cosine"
WARMUP_EPOCHS = 2
MIXED_PRECISION = True
LABEL_SMOOTHING = 0.05
MOVEMENT_LOSS_WEIGHT = 1.0
ARTIST_LOSS_WEIGHT = 0.5

# Architectures to benchmark
MODEL_CANDIDATES = [
    "resnet50",
    "convnext_tiny",
    "efficientnet_b3",
    "vit_base_patch16_224"
]
DEFAULT_MODEL = "convnext_tiny"

# Logging / evaluation artifacts
HISTORY_FILE = PROJECT_ROOT / "training_history.json"
CONFUSION_DIR = PROJECT_ROOT / "reports" / "confusion_matrices"
EMBEDDING_DIR = PROJECT_ROOT / "reports" / "embeddings"

for path in [CONFUSION_DIR, EMBEDDING_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Google Cloud helpers
GCP_BUCKET = None  # e.g. "gs://my-bucket/wikiart"
GCP_TRAINING_SCRIPT = PROJECT_ROOT / "scripts" / "train_on_gcp.sh"
