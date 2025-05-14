import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import DEVICE, MODEL_NAME
import argparse

def predict(sentence1: str, sentence2: str, model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if os.path.isdir(model_path):
        model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    elif model_path.endswith(".pth"):
        from model import get_model
        model = get_model()
        state = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

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
        prob_par = probs[1].item()
        pred = int(prob_par > probs[0].item())

    return pred, prob_par

def run_tests(model_path: str):
    print("\n== TEST PARAFRASI ==")
    
    # test_cases = [
    #     # (frase1, frase2, label_attesa)
    #     ("Roma è la capitale d'Italia.", "La capitale italiana è Roma.", 1),
    #     ("Oggi piove in città.", "Il sole splende nel cielo.", 0),
    #     ("Il cane corre nel parco.", "Un cane sta correndo in un parco.", 1),
    #     ("L'esame è stato facile.", "L'interrogazione era difficile.", 0),
    #     ("Mi piace ascoltare la musica.", "Adoro sentire le canzoni.", 1),
    #     ("Ieri sono andato al mare.", "Oggi vado in montagna.", 0),
    # ]

    test_cases = [
        ("Il sole splende alto nel cielo.", "La luna illumina la notte.", 0),
        ("Ho perso il treno stamattina.", "Sono arrivato puntuale in ufficio.", 0),
        ("Marco ha comprato una macchina nuova.", "Luca ha venduto la sua vecchia auto.", 0),
        ("La situazione è sfuggita di mano.", "Tutto è sotto controllo.", 0),
        ("Il tempo vola quando ci si diverte.", "Le ore passano lentamente durante le lezioni noiose.", 0),
        ("Ho acceso la luce.", "Ho spento la lampada.", 0),
        ("Il bambino rideva felice.", "La bambina piangeva disperata.", 0),
        ("La strada era deserta.", "C’era molta gente in piazza.", 0),
        ("Ho letto un libro interessante.", "Ho scritto una lettera importante.", 0),
        ("Il cane dorme sul divano.", "Il gatto gioca in giardino.", 0),
        ("La porta è aperta.", "La finestra è chiusa.", 0),
        ("Ho finito il lavoro.", "Devo ancora iniziare il progetto.", 0),
        ("Il film era noioso.", "Il libro era avvincente.", 0),
        ("Ho dimenticato le chiavi a casa.", "Ho ricordato di prendere il portafoglio.", 0),
        ("La pizza era calda.", "Il gelato era freddo.", 0),
        ("Era tardi quando sono arrivato.", "Sono giunto a destinazione dopo l'orario previsto.", 1),
        ("Non ho capito la domanda.", "La richiesta non mi è stata chiara.", 1),
        ("Abbiamo deciso di rimandare l’incontro.", "L’appuntamento è stato posticipato.", 1),
        ("Il bambino si è addormentato.", "Il piccolo ha preso sonno.", 1),
        ("La strada era completamente vuota.", "Non c’era nessuno lungo il percorso.", 1),
        ("Ho terminato il libro.", "Sono arrivato all’ultima pagina del romanzo.", 1),
        ("Il computer non funziona.", "Il PC è fuori uso.", 1),
        ("Mi sento stanco oggi.", "Oggi non ho molte energie.", 1),
        ("La macchina non parte.", "L’auto non si accende.", 1),
        ("Abbiamo vinto la partita.", "La nostra squadra ha ottenuto la vittoria.", 1),
    ]

    correct = 0
    for i, (s1, s2, label) in enumerate(test_cases, 1):
        pred, conf = predict(s1, s2, model_path)
        outcome = "✅ CORRETTO" if pred == label else "❌ SBAGLIATO"
        pred_str = "PARAFRASI" if pred else "NON‑PARAFRASI"
        expected_str = "PARAFRASI" if label else "NON‑PARAFRASI"
        print(f"[{i}] {outcome} | Pred: {pred_str} ({conf:.2%}) | Attesa: {expected_str}")
        print(f"  → \"{s1}\"\n     \"{s2}\"\n")
        correct += (pred == label)

    print(f"✔️  Accuracy su test set manuale: {correct}/{len(test_cases)} ({correct / len(test_cases):.2%})")

def main():
    parser = argparse.ArgumentParser(description="Paraphrase prediction")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path alla cartella di checkpoint, file .pth, o nome HF Hub"
    )
    parser.add_argument(
        "--sentence1",
        type=str,
        help="Prima frase per predizione singola"
    )
    parser.add_argument(
        "--sentence2",
        type=str,
        help="Seconda frase per predizione singola"
    )
    args = parser.parse_args()

    if args.sentence1 and args.sentence2:
        pred, conf = predict(args.sentence1, args.sentence2, args.model_path)
        pred_str = "PARAFRASI" if pred else "NON‑PARAFRASI"
        print(f"\n› Frase A: {args.sentence1!r}")
        print(f"› Frase B: {args.sentence2!r}")
        print(f"=> Model prediction: {pred_str} (P(par)= {conf:.2%})")
    else:
        run_tests(args.model_path)

if __name__ == "__main__":
    main()
