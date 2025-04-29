import torch
from transformers import AutoTokenizer
from model import get_model
from config import DEVICE, MODEL_NAME

def predict(sentence1, sentence2):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = get_model()
    model.load_state_dict(torch.load("outputs/cross_encoder_qqp.pth"))
    model.to(DEVICE)
    model.eval()

    inputs = tokenizer(
        sentence1,
        sentence2,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probs).item()

    return prediction, probs[0][prediction].item()

# Esempio:
pred, score = predict("How do I cook rice?", "How do I cook rice?")
print(pred, score)
