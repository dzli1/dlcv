import torch

# Check for CUDA first, then MPS (Apple Silicon), then fall back to CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using MPS (Apple Silicon)")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

# File Paths and Data
DATA_DIR = './artset' 
CHECKPOINT_PATH_BASE = 'best_base_model.pth'
CHECKPOINT_PATH_TUNE = 'final_best_tuned_model.pth'
HISTORY_FILE = 'training_history.json'

# Training Parameters
BATCH_SIZE = 32
NUM_EPOCHS_BASE = 20 # Initial training epochs (frozen base)
NUM_EPOCHS_TUNE = 10 # Fine-tuning epochs (unfrozen base)
BASE_LR = 1e-3 # Learning rate for the new classification head
FINE_TUNE_LR = 1e-5 # Smaller learning rate for fine-tuning entire model


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]