# model.py

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification
from config_roBERTa_mrpc import MODEL_NAME, HIDDEN_DROPOUT, DEVICE

def get_model():
    """
    Restituisce un RobertaForSequenceClassification con dropout regolato.
    """
    cfg = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        hidden_dropout_prob=HIDDEN_DROPOUT,
        attention_probs_dropout_prob=HIDDEN_DROPOUT,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        config=cfg
    )
    return model.to(DEVICE)