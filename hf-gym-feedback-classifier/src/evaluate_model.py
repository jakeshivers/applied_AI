# src/evaluate.py
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


# ========= Robust path resolution (works from any cwd) =========
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _abs_path_from_env(env_name: str, default: Path) -> Path:
    """Return absolute path from env var if set; otherwise use default.
    If env provided a relative path, resolve it against PROJECT_ROOT.
    """
    raw = os.environ.get(env_name)
    if not raw:
        return default.resolve()
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return p

MODEL_DIR = _abs_path_from_env("MODEL_DIR", PROJECT_ROOT / "artifacts" / "model")
VALID_CSV = _abs_path_from_env("VALID_CSV", PROJECT_ROOT / "data" / "valid.csv")
OUT_DIR   = _abs_path_from_env("EVAL_OUT",  PROJECT_ROOT / "artifacts" / "eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[evaluate] MODEL_DIR = {MODEL_DIR}")
print(f"[evaluate] VALID_CSV = {VALID_CSV}")
print(f"[evaluate] OUT_DIR   = {OUT_DIR}")

if not MODEL_DIR.exists():
    raise FileNotFoundError(
        f"MODEL_DIR not found: {MODEL_DIR}\n"
        "Tip: run `python src/train.py` first, or set MODEL_DIR to your saved model path."
    )
if not VALID_CSV.exists():
    raise FileNotFoundError(
        f"VALID_CSV not found: {VALID_CSV}\n"
        "Tip: set VALID_CSV=data/test.csv to score the hold-out test set."
    )


# ================== Load labels & model ==================
with open(MODEL_DIR / "labels.json", "r") as f:
    meta = json.load(f)
labels = meta["labels"]  # fixed order
label2id = {l: i for i, l in enumerate(labels)}

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).eval()


# ================== Read validation data ==================
df = pd.read_csv(VALID_CSV)
if "text" not in df.columns or "label" not in df.columns:
    raise ValueError(f"{VALID_CSV} must contain 'text' and 'label' columns. Found: {list(df.columns)}")

texts = df["text"].astype(str).tolist()
true_labels = df["label"].astype(str).tolist()


# ================== Predict in batches ==================
def predict_batch(texts, batch_size=32):
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(
                texts[i:i+batch_size],
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            logits = model(**enc).logits
            ids = logits.argmax(-1).cpu().numpy()
            preds.extend([labels[j] for j in ids])
    return preds

pred_labels = predict_batch(texts)


# ================== Metrics & artifacts ==================
y_true = np.array([label2id[l] for l in true_labels])
y_pred = np.array([label2id[l] for l in pred_labels])

# Classification report (CSV)
report_dict = classification_report(
    y_true, y_pred, target_names=labels, output_dict=True, zero_division=0
)
pd.DataFrame(report_dict).to_csv(OUT_DIR / "classification_report.csv")

# Misclassifications (CSV)
miss = pd.DataFrame({"text": texts, "true": true_labels, "pred": pred_labels})
miss = miss[miss["true"] != miss["pred"]]
miss.to_csv(OUT_DIR / "misclassifications.csv", index=False)

# Confusion matrices
cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True).clip(min=1)

def _plot_cm(mat, title, filename, normalize=False):
    fig, ax = plt.subplots(figsize=(1.2 * len(labels) + 2, 1.0 * len(labels) + 2))
    im = ax.imshow(mat, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = ".2f" if normalize else "d"
    thresh = mat.max() / 2.0 if mat.size else 0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(
                j, i, format(mat[i, j], fmt),
                ha="center", va="center",
                color="white" if mat[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)

_plot_cm(cm, "Confusion Matrix", "confusion_matrix.png", normalize=False)
_plot_cm(cm_norm, "Confusion Matrix (Normalized by True Row)", "confusion_matrix_normalized.png", normalize=True)

# Per-class Precision/Recall/F1 bar chart
per_class = pd.DataFrame(report_dict).T.loc[labels, ["precision", "recall", "f1-score"]]
fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(labels)), 4.5))
x = np.arange(len(labels))
w = 0.25
ax.bar(x - w, per_class["precision"], width=w, label="precision")
ax.bar(x,     per_class["recall"],    width=w, label="recall")
ax.bar(x + w, per_class["f1-score"],  width=w, label="f1")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("Per-class Precision / Recall / F1")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "per_class_f1.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("[evaluate] Saved:")
print(" -", OUT_DIR / "classification_report.csv")
print(" -", OUT_DIR / "misclassifications.csv")
print(" -", OUT_DIR / "confusion_matrix.png")
print(" -", OUT_DIR / "confusion_matrix_normalized.png")
print(" -", OUT_DIR / "per_class_f1.png")
