
DEVICE = "cuda"
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
EPOCHS = 5
SAVE_DIR = "outputs/"
DATASET_NAME = "qqp"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LENGTH = 64

WEIGHT_DECAY = 0.01         # weight decay for AdamW
WARMUP_STEPS = 100          # number of warmup steps for scheduler
PATIENCE = 2                # epochs to wait for improvement before early stopping
GRAD_CLIP_NORM = 1.0        # max norm for gradient clipping

# Logging
LOG_DIR = "runs/"

# Seed for reproducibility
SEED = 42
