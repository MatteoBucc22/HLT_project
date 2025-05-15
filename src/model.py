import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoModelForSequenceClassification
from config import MODEL_NAME, MAX_SEQ_LENGHT
from sentence_transformers import SentenceTransformer, SentenceTransformerModelCardData
from sentence_transformers.models import Transformer, Pooling
import torch

def get_model():
    #model = AutoModelForSequenceClassification.from_pretrained(
    #    MODEL_NAME,
    #    num_labels=2
    #)
     ## Step 1: use an existing language model
    word_embedding_model = Transformer(MODEL_NAME)

    ## Step 2: use a pool function over the token embeddings
    pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension(), 
                                pooling_mode = 'cls',
                                pooling_mode_cls_token=True, 
                                pooling_mode_mean_tokens = False)
    
    ## Join steps 1 and 2 using the modules argument
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)
