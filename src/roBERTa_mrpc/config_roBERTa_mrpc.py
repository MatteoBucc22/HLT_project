MODEL_NAME      = "roberta-base"
DATASET_NAME    = "mrpc"

MAX_LENGTH      = 64
BATCH_SIZE      = 32

LEARNING_RATE   = 2e-5
WEIGHT_DECAY    = 0.01

EPOCHS          = 6
WARMUP_RATIO    = 0.1       
WARMUP_STEPS    = None      
LR_SCHEDULER    = "linear"  
LOGGING_STEPS   = 50

HIDDEN_DROPOUT  = 0.1

SEED            = 42
DEVICE          = "cuda"    

SAVE_DIR        = "outputs/"