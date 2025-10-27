import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

def test(model, test_loader, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    correct = 0
    probabilities, predictions, targets = [], [], []

    with torch.no_grad():
        for inputs, labels, *_ in test_loader:
            inputs = inputs.to(device)
            targets.append(labels.detach().cpu().numpy())
            labels = labels.to(device)

            outputs = model(inputs)
            if isinstance(outputs, tuple):
                probs, logits = outputs
            else:
                logits = outputs
                probs = torch.softmax(logits, dim=1)

            probabilities.append(probs.detach().cpu().numpy())
            preds = torch.argmax(probs, 1)
            predictions.append(preds.detach().cpu().numpy())
            correct += preds.eq(labels.view(-1)).sum().item()

    acc = correct / len(test_loader.dataset) if len(test_loader.dataset) > 0 else 0.0

    # flatten arrays
    if len(probabilities) > 0:
        y_true = np.asarray([v for sub in targets for v in sub])
        y_prob = np.asarray([v[1] for sub in probabilities for v in sub])
        y_pred = np.asarray([np.argmax(v) for sub in probabilities for v in sub])
    else:
        y_true, y_prob, y_pred = np.array([]), np.array([]), np.array([])

    if np.any(np.isnan(y_prob)):
        print("[DEBUG] y_prob contains NaN")

    # metrics (safe)
    try:
        lr_precision, lr_recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(lr_recall, lr_precision) if len(lr_recall) > 1 else float('nan')
    except Exception as e:
        pr_auc = float('nan')
        print(f"Warning: PR curve failed ({e}), set NaN")

    try:
        roc_auc = roc_auc_score(y_true, y_prob) if y_true.size > 0 else float('nan')
    except Exception as e:
        roc_auc = float("nan")
        print(f"Warning: ROC-AUC failed ({e}), set NaN")

    cm = confusion_matrix(y_true, y_pred) if y_true.size > 0 else np.array([[]])
    return acc, roc_auc, pr_auc, cm, y_true, y_prob