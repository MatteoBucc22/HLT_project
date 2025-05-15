# src/model.py
import os
from sentence_transformers import SentenceTransformer
from config import MODEL_NAME, DEVICE

def get_siamese_model():
    # Carica SBERT come backbone siamese
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    return model