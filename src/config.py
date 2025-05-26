# config.py
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameters ottimizzati per MASSIMA accuracy su QQP validation
TOTAL_TRAIN_STEPS = 40_000  # Aumentato per più training ma con early stopping implicito
WARMUP_STEPS = 4_000       # 10% del total_steps per warm-up più graduale
BATCH_SIZE = 16            # Batch size ridotto per gradient updates più frequenti e precisi
LEARNING_RATE = 3e-4       # Learning rate ottimale per LoRA fine-tuning su QQP
EPOCHS = 5                 # Aumentato per permettere convergenza completa

# AdamW parameters ottimizzati per performance
WEIGHT_DECAY = 0.01        # Weight decay moderato per non limitare troppo l'apprendimento
GRAD_CLIP_NORM = 1.0       # Gradient clipping standard per stabilità

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

