# config.py

MODEL_NAME   = "roberta-base"
DATASET_NAME = "mrpc"

# Tokenization / dataloader
MAX_LENGTH   = 64
BATCH_SIZE   = 32

# Training
LEARNING_RATE   = 2e-5           # leggermente più basso per stabilità
WEIGHT_DECAY    = 0.01           # per regolarizzare
EPOCHS          = 6              # puoi spingere fino a 6–8 con early stopping
WARMUP_RATIO    = 0.1            # frazione di passi per warm-up
LR_SCHEDULER    = "linear"       # oppure "linear", "step"
WARMUP_STEPS    = None           # se None, calcolato come WARMUP_RATIO * total_steps

# Dropout
HIDDEN_DROPOUT  = 0.1            # aumenta leggermente il dropout interno
LORA_DROPOUT    = 0.05          # se usi LoRA

# Sperimentazione
SEED            = 42
DEVICE          = "cuda"

# Output / logging
SAVE_DIR        = "outputs/"
LOGGING_STEPS   = 50             # per stampare loss / lr ogni n batch
