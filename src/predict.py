# predict.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoTokenizer
from model import get_model
from config import DEVICE, MODEL_NAME

def predict(sentence1: str, sentence2: str, checkpoint_path: str):
    """
    Restituisce (pred, conf) dove pred è 1 se parafrasi, 0 altrimenti,
    e conf è la probabilità associata.
    """
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # modello + caricamento weights
    model = get_model()
    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()

    # inferenza
    inputs = tokenizer(
        sentence1,
        sentence2,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        pred = int(probs[1] > probs[0])
        conf = probs[pred].item()

    return pred, conf


def test_pair(sentence1: str, sentence2: str, is_para: int, checkpoint_path: str):
    pred, conf = predict(sentence1, sentence2, checkpoint_path)
    gt = "PARAFRASI" if is_para else "NON-PARAFRASI"
    pr = "PARAFRASI" if pred else "NON-PARAFRASI"
    print(f"A: {sentence1!r}\nB: {sentence2!r}")
    print(f"GT: {gt} — Pred: {pr} (conf {conf:.2%})\n")


if __name__ == "__main__":
    ckpt = "outputs/roberta-base-mrpc-full_YYYYMMDD_HHMMSS.pth"  # sostituisci con il tuo .pth
    test_pair(
        "How do I cook rice?",
        "Best way to cook rice?",
        is_para=1,
        checkpoint_path=ckpt
    )
    test_pair(
        "How do I cook rice?",
        "How old are you?",
        is_para=0,
        checkpoint_path=ckpt
    )
