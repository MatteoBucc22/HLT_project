import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from data_loader import get_datasets
from config import DEVICE, BATCH_SIZE, DATASET_NAME

def load_model(repo_id):
    print(f"Loading model from {repo_id}")
    model = AutoModelForSequenceClassification.from_pretrained(repo_id).to(DEVICE)
    model.eval()
    return model

def ensemble_predict(model1, model2, dataloader, weight1=0.5, weight2=0.5):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"].to(DEVICE)
            inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != "labels"}

            outputs1 = model1(**inputs)
            outputs2 = model2(**inputs)

            # Weighted average logits
            logits_ensemble = weight1 * outputs1.logits + weight2 * outputs2.logits
            preds = torch.argmax(logits_ensemble, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return all_preds, all_labels

def save_predictions(preds, labels, file_path="ensemble_predictions.txt"):
    with open(file_path, "w") as f:
        f.write("Prediction\tLabel\n")
        for p, l in zip(preds, labels):
            f.write(f"{p}\t{l}\n")
    print(f"Predizioni salvate in {file_path}")

def main():
    model1_repo = "MatteoBucc/passphrase-identification-roberta-base-qqp-embeddings-20250515_135729"
    model2_repo = "MatteoBucc/passphrase-identification-roberta-base-mrpc-best"

    model1 = load_model(model1_repo)
    model2 = load_model(model2_repo)

    dataset = get_datasets()
    val_loader = DataLoader(dataset["validation"], batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # Cambia i pesi per dare più importanza a un modello (deve sommare a 1)
    weight1 = 0.7  
    weight2 = 0.3

    preds, labels = ensemble_predict(model1, model2, val_loader, weight1=weight1, weight2=weight2)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)

    print(f"Ensemble validation results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    save_predictions(preds, labels)

if __name__ == "__main__":
    main()
