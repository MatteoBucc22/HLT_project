import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import DEVICE, MODEL_NAME
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from model import get_model  # Se hai un metodo per costruire il tuo modello

def predict(sentence1: str, sentence2: str, model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)  # Usa il modello di base per il tokenizer

    # Carica il modello con i pesi salvati nel file .pth
    model = get_model()  # O costruisci il modello in base alla tua architettura
    model.load_state_dict(torch.load(model_path))  # Carica i pesi dal file .pth
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
    parser.add_argument("--model_path", type=str, required=True, help="Path al modello fine-tunato")
    args = parser.parse_args()

    pred, conf = predict(args.sentence1, args.sentence2, args.model_path)
    pred_str = "PARAFRASI" if pred else "NON‑PARAFRASI"

    print(f"› Frase A: {args.sentence1!r}")
    print(f"› Frase B: {args.sentence2!r}\n")
    print(f"=> Model prediction: {pred_str} (conf {conf:.2%})")

if __name__ == "__main__":
    main()
