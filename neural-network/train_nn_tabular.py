# train_nn_tabular.py
import os, math, time, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 40
LR = 1e-3
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# --------- 1) Data ----------
data = load_breast_cancer()
X = data.data.astype(np.float32)
y = data.target.astype(np.float32)

# Standardize features (important for NNs)
scaler = StandardScaler()
X = scaler.fit_transform(X).astype(np.float32)

X_t = torch.tensor(X)
y_t = torch.tensor(y).unsqueeze(1)  # shape (N,1)

ds = TensorDataset(X_t, y_t)

# train / val / test = 70/15/15
n = len(ds)
n_train = int(0.7 * n)
n_val = int(0.15 * n)
n_test = n - n_train - n_val
train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(SEED))

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

input_dim = X.shape[1]


class LogisticRegression(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.lin = nn.Linear(d_in, 1)
    def forward(self, x):
        return self.lin(x)

class MLP1(nn.Module):
    # One hidden layer
    def __init__(self, d_in, d_h=32, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_h),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(d_h, 1)
        )
    def forward(self, x): return self.net(x)

class DeepMLP(nn.Module):
    # Deeper net: two hidden layers
    def __init__(self, d_in, d_h1=64, d_h2=32, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_h1), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_h1, d_h2), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_h2, 1)
        )
    def forward(self, x): return self.net(x)


def train_one_epoch(model, loader, optim, loss_fn):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optim.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optim.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_logits, all_targets = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        logits = model(xb)
        all_logits.append(logits.cpu())
        all_targets.append(yb)
    logits = torch.cat(all_logits).squeeze(1)
    targets = torch.cat(all_targets).squeeze(1)
    probs = torch.sigmoid(logits).numpy()
    preds = (probs >= 0.5).astype(np.int32)
    y_true = targets.numpy().astype(np.int32)

    acc = accuracy_score(y_true, preds)
    try:
        roc = roc_auc_score(y_true, probs)
    except ValueError:
        roc = float("nan")
    precision, recall, _ = precision_recall_curve(y_true, probs)
    pr_auc = auc(recall, precision)
    return {"acc": acc, "roc_auc": roc, "pr_auc": pr_auc, "probs": probs, "y_true": y_true}

def train_model(model, train_dl, val_dl, epochs=EPOCHS, lr=LR, name="model"):
    model.to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val = -1.0
    history = {"train_loss": [], "val_roc": [], "val_pr": []}
    os.makedirs("artifacts", exist_ok=True)
    path = f"artifacts/{name}.pt"

    patience, patience_ctr = 6, 0
    for epoch in range(1, epochs+1):
        tr_loss = train_one_epoch(model, train_dl, optim, loss_fn)
        val_metrics = evaluate(model, val_dl)
        history["train_loss"].append(tr_loss)
        history["val_roc"].append(val_metrics["roc_auc"])
        history["val_pr"].append(val_metrics["pr_auc"])

        msg = f"[{name}] Epoch {epoch:02d} | loss {tr_loss:.4f} | val ROC {val_metrics['roc_auc']:.4f} | val PR {val_metrics['pr_auc']:.4f}"
        print(msg)

        score = (val_metrics["roc_auc"] if not math.isnan(val_metrics["roc_auc"]) else 0.0) + val_metrics["pr_auc"]
        if score > best_val:
            best_val = score
            torch.save(model.state_dict(), path)
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"[{name}] Early stopping at epoch {epoch}")
                break

    with open(f"artifacts/{name}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    return path, history

# --------- 4) Train the three models ----------
models = {
    "logreg": LogisticRegression(input_dim),
    "mlp1": MLP1(input_dim),
    "deep": DeepMLP(input_dim)
}
paths = {}
for name, mdl in models.items():
    p, _ = train_model(mdl, train_dl, val_dl, name=name)
    paths[name] = p

# --------- 5) Evaluate on test set & plot ----------
@torch.no_grad()
def load_and_eval(path):
    state_dict = torch.load(path, map_location=DEVICE)
    # figure out which arch from filename
    if "logreg" in path: model = LogisticRegression(input_dim)
    elif "mlp1" in path: model = MLP1(input_dim)
    else: model = DeepMLP(input_dim)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    return evaluate(model, test_dl)

results = {}
for name, p in paths.items():
    results[name] = load_and_eval(p)
    print(f"[{name}] Test ACC: {results[name]['acc']:.4f} | ROC-AUC: {results[name]['roc_auc']:.4f} | PR-AUC: {results[name]['pr_auc']:.4f}")

# Plot ROC curves
plt.figure()
for name, r in results.items():
    fpr, tpr, _ = roc_curve(r["y_true"], r["probs"])
    plt.plot(fpr, tpr, label=f"{name} (AUC={r['roc_auc']:.3f})")
plt.plot([0,1],[0,1],"k--", alpha=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.tight_layout()
plt.savefig("artifacts/roc_curves.png", dpi=140)

# Plot PR curves
plt.figure()
for name, r in results.items():
    precision, recall, _ = precision_recall_curve(r["y_true"], r["probs"])
    plt.plot(recall, precision, label=f"{name} (AUC={r['pr_auc']:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves")
plt.legend()
plt.tight_layout()
plt.savefig("artifacts/pr_curves.png", dpi=140)

print("Saved: artifacts/roc_curves.png, artifacts/pr_curves.png")

def choose_threshold(y_true, y_prob, mode="target_precision", target=0.90):
    precision, recall, thresh = precision_recall_curve(y_true, y_prob)
    thresh = np.r_[0.0, thresh]  # align shapes

    if mode == "target_precision":
        idx = np.where(precision >= target)[0]
        if len(idx) == 0:
            # fall back to best F1 if target not reachable
            f1s = [f1_score(y_true, (y_prob >= t).astype(int)) for t in thresh]
            best_i = int(np.argmax(f1s))
            return float(thresh[best_i]), {"precision": float(precision[best_i]), "recall": float(recall[best_i])}
        i = int(idx[0])
        return float(thresh[i]), {"precision": float(precision[i]), "recall": float(recall[i])}

    elif mode == "max_f1":
        f1s = [f1_score(y_true, (y_prob >= t).astype(int)) for t in thresh]
        best_i = int(np.argmax(f1s))
        return float(thresh[best_i]), {"precision": float(precision[best_i]), "recall": float(recall[best_i])}

    else:
        raise ValueError("mode must be 'target_precision' or 'max_f1'")

val = evaluate(model, val_dl)
t_star, stats = choose_threshold(val["y_true"], val["probs"], mode="target_precision", target=0.90)
print(f"Chosen threshold={t_star:.3f} at precision≈{stats['precision']:.2f}, recall≈{stats['recall']:.2f}")

def summarize_at_threshold(y_true, y_prob, threshold=0.5, title=""):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n=== {title} @ threshold={threshold:.2f} ===")
    print(cm)
    print(classification_report(y_true, y_pred, digits=3))
    return cm

# example using the chosen threshold for the 'deep' model
deep_res = results["deep"]
t_star, _ = choose_threshold(deep_res["y_true"], deep_res["probs"], mode="max_f1")
summarize_at_threshold(deep_res["y_true"], deep_res["probs"], t_star, title="DeepMLP (test)")
