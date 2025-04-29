from transformers import AutoModelForSequenceClassification
from config import MODEL_NAME

def get_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )
    model = get_model()
    print(f"Model is loaded on device: {next(model.parameters()).device}")
    return model.to("cuda")  # Assicurati di spostarlo esplicitamente su CUDA
