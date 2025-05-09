import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoTokenizer
from peft import PeftModel
from model import get_model
from config import DEVICE, MODEL_NAME
import argparse

def predict(sentence1: str, sentence2: str):
    """Restituisce (pred, conf) dove pred è 1 se parafrasi, 0 altrimenti, 
    e conf è la probabilità associata."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    base_model = get_model()
    model = PeftModel.from_pretrained(base_model, "outputs/lora_adapter")
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
        probs = torch.softmax(outputs.logits, dim=1)[0]
        pred = int(probs[1] > probs[0])
        conf = probs[pred].item()

    return pred, conf

def main():
    parser = argparse.ArgumentParser(description="Paraphrase prediction")
    parser.add_argument("sentence1", type=str, help="Prima frase")
    parser.add_argument("sentence2", type=str, help="Seconda frase")
    args = parser.parse_args()

    pred, conf = predict(args.sentence1, args.sentence2)
    pred_str = "PARAFRASI" if pred else "NON‑PARAFRASI"

    print(f"› Frase A: {args.sentence1!r}")
    print(f"› Frase B: {args.sentence2!r}\n")
    print(f"=> Model prediction: {pred_str} (conf {conf:.2%})")

if __name__ == "__main__":
    main()
