import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoTokenizer
from peft import PeftModel
from model import get_model
from config import DEVICE, MODEL_NAME

def predict(sentence1: str, sentence2: str, adapter_path: str = "outputs/lora_adapter"):
    """Restituisce (pred, conf) dove pred è 1 se parafrasi, 0 altrimenti, 
    e conf è la probabilità associata."""
    # 1) tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 2) modello base + LoRA adapter
    base_model = get_model()
    
    # Controllo se il path dell'adapter esiste
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"❌ Adapter path non trovato: {adapter_path}")
    
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.to(DEVICE)
    model.eval()

    # 3) tokenizzazione
    inputs = tokenizer(
        sentence1,
        sentence2,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(DEVICE)

    # 4) inferenza
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        # scelgo classe con probabilità maggiore
        pred = int(probs[1] > probs[0])
        conf = probs[pred].item()

    return pred, conf

def test_pair(sentence1: str, sentence2: str, is_para: int, adapter_path: str = "outputs/lora_adapter"):
    """Stampa frase, ground truth e predizione con confidenza."""
    pred, conf = predict(sentence1, sentence2, adapter_path)
    label_str = "PARAFRASI" if is_para else "NON‑PARAFRASI"
    pred_str  = "PARAFRASI" if pred   else "NON‑PARAFRASI"

    print(f"› Frase A: {sentence1!r}")
    print(f"› Frase B: {sentence2!r}\n")
    print(f"=> Ground truth: {label_str}")
    print(f"=> Model       : {pred_str} (conf {conf:.2%})\n")

if __name__ == "__main__":
    # Esempi di test
    test_pair(
        "How do I cook rice?",
        "Best way to cook rice?",
        is_para=1
    )
    test_pair(
        "How do I cook rice?",
        "How old are you?",
        is_para=0
    )
