# config.py
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameters from MiniLM paper
TOTAL_TRAIN_STEPS = 400_000
WARMUP_STEPS = 4_000
BATCH_SIZE = 1024
LEARNING_RATE = 5e-4
EPOCHS = 10  # approximate given step budget

# AdamW parameters
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0

# Tokenization
MAX_LENGTH = 128

# Saving & dataset
SAVE_DIR = 'outputs/'
DATASET_NAME = 'qqp'
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

# Logging & seed
LOG_DIR = 'runs/'
SEED = 42

# Embedding generation
EMB_SAVE_SUBDIR = 'validation_embeddings'

