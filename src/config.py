# config.py
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameters ottimizzati per QQP classification con LoRA
TOTAL_TRAIN_STEPS = 10_000  # Ridotto per evitare overfitting
WARMUP_STEPS = 1_000       # 10% del total_steps per warm-up graduale
BATCH_SIZE = 32            # Batch size più piccolo per training più stabile
LEARNING_RATE = 2e-4       # Learning rate più conservativo per LoRA
EPOCHS = 5                 # Meno epoch per evitare overfitting

# AdamW parameters ottimizzati
WEIGHT_DECAY = 0.1         # Weight decay più aggressivo per regularization
GRAD_CLIP_NORM = 0.5       # Gradient clipping più stretto

# Tokenization ottimizzata per QQP
MAX_LENGTH = 256           # Lunghezza maggiore per catturare meglio le parafrasi lunghe

# Saving & dataset
SAVE_DIR = 'outputs/'
DATASET_NAME = 'qqp'
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

# Logging & seed
LOG_DIR = 'runs/'
SEED = 42

# Embedding generation
EMB_SAVE_SUBDIR = 'validation_embeddings'

