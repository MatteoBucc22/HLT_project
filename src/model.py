import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoConfig, AutoModelForSequenceClassification
from config import MODEL_NAME, HIDDEN_DROPOUT


def get_model(
              hidden_dropout_prob: float = HIDDEN_DROPOUT):
    """
    Carica un modello di sequence classification con dropout personalizzato.
    Args:
        model_name: identificatore del modello pre-addestrato su HF Hub.
        hidden_dropout_prob: valore di dropout da applicare allo strato nascosto.
    Returns:
        Instanza di AutoModelForSequenceClassification non spostata su device.
    """
    # Carica la configurazione con dropout specificato
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        hidden_dropout_prob=hidden_dropout_prob
    )

    # Carica il modello con la configurazione modificata
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        config=config
    )

    return model
