import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoModelForSequenceClassification
from config import MODEL_NAME
import torch

def get_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2 
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)
