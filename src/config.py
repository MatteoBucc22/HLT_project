DATASET_NAME = "mrpc"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Tokenization / DataLoader
MAX_LENGTH      = 64
BATCH_SIZE      = 32

# Training
LEARNING_RATE   = 2e-5
WEIGHT_DECAY    = 0.01

EPOCHS          = 6
WARMUP_RATIO    = 0.1       # frazione di total_steps per warm-up
WARMUP_STEPS    = None      # se None, calcolato da WARMUP_RATIO
LR_SCHEDULER    = "linear"  # "linear", "cosine", "step"
LOGGING_STEPS   = 50

# Dropout interno di RoBERTa
HIDDEN_DROPOUT  = 0.1

# Misc
SEED            = 42
DEVICE          = "cuda"    # o "cpu"

# Outputs
SAVE_DIR        = "outputs/"