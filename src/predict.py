import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import DEVICE, MODEL_NAME
import argparse

def predict(sentence1: str, sentence2: str, model_path: str):
    """
    Esegue inferenza su coppia di frasi senza soglia esterna: decide sulla base
    del confronto diretto tra le due probabilità (classe 0 vs classe 1).
    """
    # 1) Carica tokenizer dal modello di base
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 2) Carica il modello (da cartella locale, .pth o HF Hub)
    if os.path.isdir(model_path):
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, local_files_only=True
        )
    elif model_path.endswith(".pth"):
        from model import get_model
        model = get_model()
        state = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

    model.to(DEVICE)
    model.eval()

    # 3) Inferenza
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
        # Confronto diretto delle probabilità
        prob_nonpar, prob_par = probs[0].item(), probs[1].item()
        pred = int(prob_par > prob_nonpar)
        conf = prob_par if pred == 1 else prob_nonpar

    return pred, conf

def main():
    parser = argparse.ArgumentParser(description="Paraphrase prediction")
    parser.add_argument("sentence1", type=str, help="Prima frase")
    parser.add_argument("sentence2", type=str, help="Seconda frase")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path alla cartella di checkpoint, file .pth, o nome HF Hub"
    )
    args = parser.parse_args()

    pred, conf = predict(
        args.sentence1,
        args.sentence2,
        args.model_path
    )
    pred_str = "PARAFRASI" if pred else "NON‑PARAFRASI"

    print(f"› Frase A: {args.sentence1!r}")
    print(f"› Frase B: {args.sentence2!r}\n")
    print(f"=> Model prediction: {pred_str} (conf={conf:.2%})")

if __name__ == "__main__":
    main()
