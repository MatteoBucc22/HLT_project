DATASET_NAME = "mrpc"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Tokenization / DataLoader - Ottimizzato per MRPC
MAX_LENGTH      = 128       # Aumentato per catturare meglio le frasi lunghe
BATCH_SIZE      = 16        # Ridotto per gradienti più stabili

# Training - Parametri ottimizzati per migliore accuracy
LEARNING_RATE   = 1e-5      # Learning rate più basso per fine-tuning stabile
WEIGHT_DECAY    = 0.01
GRADIENT_CLIPPING = 1.0     # Aggiunto per stabilità

EPOCHS          = 8         # Aumentato per migliore convergenza
WARMUP_RATIO    = 0.1       # frazione di total_steps per warm-up
WARMUP_STEPS    = None      # se None, calcolato da WARMUP_RATIO
LR_SCHEDULER    = "cosine"  # Cosine annealing per migliore convergenza
LOGGING_STEPS   = 25        # Logging più frequente

# Dropout ottimizzato per prevenire overfitting
HIDDEN_DROPOUT  = 0.2       # Aumentato per dataset piccolo
ATTENTION_DROPOUT = 0.1     # Aggiunto per robustezza

# Early stopping
PATIENCE        = 3         # Early stopping per evitare overfitting
MIN_DELTA       = 0.001     # Miglioramento minimo richiesto

# Misc
SEED            = 42
DEVICE          = "cuda"    # o "cpu"

# Outputs
SAVE_DIR        = "outputs/"