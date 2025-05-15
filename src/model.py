# src/model.py
import os
from sentence_transformers import SentenceTransformer
from config import MODEL_NAME, DEVICE

def get_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2 
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)
