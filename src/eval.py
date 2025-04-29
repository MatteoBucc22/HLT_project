import torch
from sklearn.metrics import accuracy_score, f1_score

def evaluate(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            # Sposta i tensori al dispositivo
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["is_duplicate"].to(model.device)  # Assicurati che siano etichette di tipo long

            # Ottieni gli output del modello
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Previsioni: argmax per ottenere la classe predetta
            preds = outputs.logits.argmax(dim=1)

            # Aggiungi le previsioni e le etichette
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calcola la precisione e il punteggio F1
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    # Stampa i risultati
    print(f"Validation Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")
