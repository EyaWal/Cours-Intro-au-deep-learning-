import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from train_optimizers import train_model, device, test_loader

def evaluate_model(model, test_loader):
    model.eval()
    all_targets = []
    all_preds_probs = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch["features"].to(device), batch["labels"].to(device)
            outputs = model(inputs)
            all_targets.extend(targets.cpu().numpy())
            all_preds_probs.extend(outputs.cpu().numpy())

    all_targets = np.array(all_targets)
    all_preds_probs = np.array(all_preds_probs)
    all_preds_classes = (all_preds_probs > 0.5).astype(int)

    precision = precision_score(all_targets, all_preds_classes)
    recall = recall_score(all_targets, all_preds_classes)
    f1 = f1_score(all_targets, all_preds_classes)
    auc = roc_auc_score(all_targets, all_preds_probs)

    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

if __name__ == "__main__":
    best_model = train_model("Adam", learning_rate=0.001)
    evaluate_model(best_model, test_loader)