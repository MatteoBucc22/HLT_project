import torch
from transformers import AutoTokenizer
from model import get_model
from peft import PeftModel
from config import DEVICE, MODEL_NAME

def predict(sentence1, sentence2):
    # Carica tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Carica modello base
    base_model = get_model()

    # Applica LoRA adapter salvato
    model = PeftModel.from_pretrained(base_model, "outputs/lora_adapter")
    model.to(DEVICE)
    model.eval()

    # Tokenizza coppia di frasi
    inputs = tokenizer(
        sentence1,
        sentence2,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(DEVICE)

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probs).item()

    return prediction, probs[0][prediction].item()

# Esempio:
pred, score = predict("How do I cook rice?", "How do I cook rice?")
print(pred, score)
