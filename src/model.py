# model.py

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoConfig, AutoModelForSequenceClassification
from config import MODEL_NAME, HIDDEN_DROPOUT, ATTENTION_DROPOUT


def get_model(hidden_dropout_prob: float = HIDDEN_DROPOUT,
              attention_dropout_prob: float = ATTENTION_DROPOUT):
    """
    Carica un modello di sequence classification con dropout ottimizzato.
    """
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_dropout_prob,
        classifier_dropout=hidden_dropout_prob,  # Dropout nel classifier
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        config=config
    )
    return model