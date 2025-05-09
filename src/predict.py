import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import DEVICE, MODEL_NAME
import argparse

def predict(sentence1: str, sentence2: str, model_path: str):
    # 1) tokenizer: sempre dal modello base
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 2) modello: dal checkpoint salvato in model_path
    if os.path.isdir(model_path):
        # Carica config.json + safetensors (o pytorch_model.bin)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    elif model_path.endswith(".pth"):
        # File .pth: usa la tua architettura e carica lo state dict
        from model import get_model
        model = get_model()
        state = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state)
    else:
        # Nome Hugging Face Hub
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

    model.to(DEVICE)
    model.eval()

    # 3) inferenza
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
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path alla cartella di checkpoint (config.json + safetensors) o file .pth, o nome HF Hub"
    )
    args = parser.parse_args()

    pred, conf = predict(args.sentence1, args.sentence2, args.model_path)
    pred_str = "PARAFRASI" if pred else "NON‑PARAFRASI"

    print(f"› Frase A: {args.sentence1!r}")
    print(f"› Frase B: {args.sentence2!r}\n")
    print(f"=> Model prediction: {pred_str} (conf {conf:.2%})")

if __name__ == "__main__":
    main()
