# config.py

DATASET_NAME   = "mrpc"
MODEL_NAME     = "sentence-transformers/all-MiniLM-L6-v2"

# Tokenization / DataLoader
MAX_LENGTH     = 64
BATCH_SIZE     = 32
NUM_WORKERS    = 4
PIN_MEMORY     = True

# Training
LEARNING_RATE  = 2e-5
WEIGHT_DECAY   = 0.01
EPOCHS         = 10

# Scheduler & warm-up
LR_SCHEDULER   = "cosine"   # "linear" o "cosine"
WARMUP_RATIO   = 0.2        # frazione dei total_steps
WARMUP_STEPS   = None       # se non None, sovrascrive WARMUP_RATIO

# Gradient accumulation: simula batch grandi
ACCUM_STEPS    = 2

# Label smoothing
LABEL_SMOOTHING = 0.1

# Early stopping
EARLY_STOPPING_PATIENCE = 3

# Logging
LOGGING_STEPS  = 50

# Dropout interno del Transformer
HIDDEN_DROPOUT = 0.2        # puoi provare anche 0.1 -> 0.3

# Misc\ nSEED           = 42
DEVICE         = "cuda"     # o "cpu"

# Outputs
SAVE_DIR       = "outputs/"

# hf_utils repo prefix
HF_REPO_PREFIX = "MatteoBucc/passphrase-identification"

SEED         = 42