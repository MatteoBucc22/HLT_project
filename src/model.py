# model.py

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoConfig, AutoModelForSequenceClassification
from config import MODEL_NAME, HIDDEN_DROPOUT


def get_model(
              hidden_dropout_prob: float = HIDDEN_DROPOUT):
    """
    Carica un modello di sequence classification con dropout personalizzato.
    """
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        hidden_dropout_prob=hidden_dropout_prob
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        config=config
    )
    return model