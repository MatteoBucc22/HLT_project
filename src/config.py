DATASET_NAME = "mrpc"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Tokenization / DataLoader
MAX_LENGTH      = 512       # Aumentato per MRPC che ha frasi più lunghe
BATCH_SIZE      = 8         # Ridotto per consentire gradienti più stabili

# Training - Parametri ottimizzati per MRPC
LEARNING_RATE   = 2e-5      # Learning rate più conservativo per fine-tuning
WEIGHT_DECAY    = 0.01      
GRADIENT_CLIPPING = 1.0     # Manteniamo per stabilità

EPOCHS          = 6         # Aumentato per permettere migliore convergenza
WARMUP_RATIO    = 0.1       # Warmup più lungo per stabilità iniziale
WARMUP_STEPS    = None      # se None, calcolato da WARMUP_RATIO
LR_SCHEDULER    = "linear"  # Linear decay dopo warmup (più stabile)
LOGGING_STEPS   = 10        # Logging più frequente per monitoraggio

# Dropout ottimizzato per prevenire overfitting
HIDDEN_DROPOUT  = 0.3       # Aumentato per dataset piccolo
ATTENTION_DROPOUT = 0.1     # Manteniamo per robustezza

# Early stopping più paziente
PATIENCE        = 4         # Più paziente per permettere convergenza
MIN_DELTA       = 0.0005    # Soglia più bassa per miglioramenti

# Misc
SEED            = 42
DEVICE          = "cuda"    # o "cpu"

# Outputs
SAVE_DIR        = "outputs/"