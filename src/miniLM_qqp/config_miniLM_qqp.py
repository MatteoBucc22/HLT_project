import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TOTAL_TRAIN_STEPS = 40_000  
WARMUP_STEPS = 4_000       
BATCH_SIZE = 16            
LEARNING_RATE = 3e-4       
EPOCHS = 5                

WEIGHT_DECAY = 0.01       
GRAD_CLIP_NORM = 1.0       

MAX_LENGTH = 256          

SAVE_DIR = 'outputs/'
DATASET_NAME = 'qqp'
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

LOG_DIR = 'runs/'
SEED = 42

EMB_SAVE_SUBDIR = 'validation_embeddings'