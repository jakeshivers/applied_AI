# src/train.py
import os
from pathlib import Path
import json
import pandas as pd

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import numpy as np
import evaluate

import mlflow, mlflow.transformers

# (MLflow experiment/run logic moved to after all variables are defined)





# ---------- robust path resolution ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _abs_path_from_env(env_name: str, default: Path) -> Path:
    raw = os.environ.get(env_name)
    if not raw:
        return default.resolve()
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return p

MODEL_NAME = os.environ.get("MODEL_NAME", "distilbert-base-uncased")
TRAIN_CSV  = _abs_path_from_env("TRAIN_CSV", PROJECT_ROOT / "data" / "train.csv")
VALID_CSV  = _abs_path_from_env("VALID_CSV", PROJECT_ROOT / "data" / "valid.csv")
OUT_DIR    = _abs_path_from_env("OUT_DIR",   PROJECT_ROOT / "artifacts" / "model")
SEED       = int(os.environ.get("SEED", "42"))
MAX_LEN    = int(os.environ.get("MAX_LEN", "256"))

OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[train] MODEL_NAME = {MODEL_NAME}")
print(f"[train] TRAIN_CSV  = {TRAIN_CSV}")
print(f"[train] VALID_CSV  = {VALID_CSV}")
print(f"[train] OUT_DIR    = {OUT_DIR}")

if not TRAIN_CSV.exists():
    raise FileNotFoundError(f"TRAIN_CSV not found: {TRAIN_CSV}")
if not VALID_CSV.exists():
    raise FileNotFoundError(f"VALID_CSV not found: {VALID_CSV}")

# ---------- load data & labels ----------
train_df = pd.read_csv(TRAIN_CSV)
valid_df = pd.read_csv(VALID_CSV)

if "text" not in train_df.columns or "label" not in train_df.columns:
    raise ValueError(f"{TRAIN_CSV} must contain 'text' and 'label' columns.")
if "text" not in valid_df.columns or "label" not in valid_df.columns:
    raise ValueError(f"{VALID_CSV} must contain 'text' and 'label' columns.")

labels = sorted(train_df["label"].astype(str).unique().tolist())
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

train_df = train_df.copy()
valid_df = valid_df.copy()
train_df["labels"] = train_df["label"].astype(str).map(label2id)
valid_df["labels"] = valid_df["label"].astype(str).map(label2id)

train_ds = Dataset.from_pandas(train_df[["text", "labels"]], preserve_index=False)
valid_ds = Dataset.from_pandas(valid_df[["text", "labels"]], preserve_index=False)

# ---------- tokenizer & model ----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tok(examples):
    return tokenizer(examples["text"], truncation=True, max_length=MAX_LEN)

train_ds = train_ds.map(tok, batched=True)
valid_ds = valid_ds.map(tok, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(labels), id2label=id2label, label2id=label2id
)

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, y = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc_result = accuracy.compute(predictions=preds, references=y)
    f1_result = f1.compute(predictions=preds, references=y, average="macro")
    return {
        "accuracy": acc_result["accuracy"] if acc_result and "accuracy" in acc_result else 0.0,
        "f1_macro": f1_result["f1"] if f1_result and "f1" in f1_result else 0.0,
    }

args = TrainingArguments(
    output_dir=str(OUT_DIR),
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=int(os.environ.get("EPOCHS", "3")),
    weight_decay=0.01,
    seed=SEED,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    logging_steps=50,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(str(OUT_DIR))
tokenizer.save_pretrained(str(OUT_DIR))

with open(OUT_DIR / "labels.json", "w") as f:
    json.dump({"labels": labels, "label2id": label2id, "id2label": id2label}, f, indent=2)

print("[train] Training complete. Artifacts saved to:", OUT_DIR)
mlflow.set_experiment("gym-feedback")
with mlflow.start_run(run_name="distilbert"):
    mlflow.log_params({
        "model_name": MODEL_NAME,
        "epochs": int(os.environ.get("EPOCHS", "3")),
        "seed": SEED,
        "max_len": MAX_LEN,
        "num_labels": len(labels),
    })
    trainer.train()
    eval_metrics = trainer.evaluate()
    for k, v in eval_metrics.items():
        try:
            mlflow.log_metric(k, float(v))
        except Exception:
            pass
    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))
    with open(OUT_DIR / "labels.json", "w") as f:
        json.dump({"labels": labels, "label2id": label2id, "id2label": id2label}, f, indent=2)
    mlflow.log_artifacts(str(OUT_DIR), artifact_path="model_artifacts")

    mlflow.transformers.log_model(
        transformers_model={"model": model, "tokenizer": tokenizer},
        artifact_path="transformer",
        input_example={"text": "I was double charged this month"},
        registered_model_name="gym-feedback-classifier",  # local registry
    )

print("[train] Training complete. Artifacts saved to:", OUT_DIR)
