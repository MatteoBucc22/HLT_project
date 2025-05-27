import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoModelForSequenceClassification
from config_miniLM_qqp import MODEL_NAME, DEVICE

def get_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )
    return model.to(DEVICE)