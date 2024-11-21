import torch
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Ensure directories exist
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Audio processing parameters
TARGET_SR = 44100
N_MELS = 64
N_FFT = 2048
HOP_LENGTH = 512
DURATION = 30
MAX_FILE_SIZE = 30 * TARGET_SR

# Model parameters
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.1
NUM_EMOTIONAL_FEATURES = 22

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
EARLY_STOPPING_PATIENCE = 5
NUM_WORKERS = 4
# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Feature names
VALENCE_FEATURES = [
    "V_mean",
    "V_std",
    "V_median",
    "V_mode",
    "V_kurtosis",
    "V_skewness",
    "V_variation",
    "V_quartile1",
    "V_quartile3",
    "V_max",
    "V_min",
]

AROUSAL_FEATURES = [
    "A_mean",
    "A_std",
    "A_median",
    "A_mode",
    "A_kurtosis",
    "A_skewness",
    "A_variation",
    "A_quartile1",
    "A_quartile3",
    "A_max",
    "A_min",
]
