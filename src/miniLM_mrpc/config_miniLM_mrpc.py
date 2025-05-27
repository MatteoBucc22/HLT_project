DATASET_NAME = "mrpc"
MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"

MAX_LENGTH      = 128       
BATCH_SIZE      = 12        

LEARNING_RATE   = 8e-6      
WEIGHT_DECAY    = 0.01
GRADIENT_CLIPPING = 1.0    

EPOCHS          = 8        
WARMUP_RATIO    = 0.1       
WARMUP_STEPS    = None      
LR_SCHEDULER    = "cosine"  
LOGGING_STEPS   = 25        

HIDDEN_DROPOUT  = 0.2      
ATTENTION_DROPOUT = 0.1   

PATIENCE        = 3        
MIN_DELTA       = 0.001     

SEED            = 42
DEVICE          = "cuda"    

SAVE_DIR        = "outputs/"