# src/batch_predict.py
import os
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

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

MODEL_DIR = _abs_path_from_env("MODEL_DIR", PROJECT_ROOT / "artifacts" / "model")
IN_CSV    = _abs_path_from_env("IN_CSV",  PROJECT_ROOT / "data" / "inference.csv")
OUT_CSV   = _abs_path_from_env("OUT_CSV", PROJECT_ROOT / "artifacts" / "predictions.csv")

if not MODEL_DIR.exists():
    raise FileNotFoundError(f"MODEL_DIR not found: {MODEL_DIR}. Train first or set MODEL_DIR.")

# Fallback if inference.csv missing: use valid.csv texts
if not IN_CSV.exists():
    fallback = PROJECT_ROOT / "data" / "valid.csv"
    if not fallback.exists():
        raise FileNotFoundError(f"IN_CSV not found: {IN_CSV} and fallback not found: {fallback}")
    df = pd.read_csv(fallback)
    df = df[["text"]].copy()
else:
    df = pd.read_csv(IN_CSV)
    if "text" not in df.columns:
        raise ValueError(f"Input CSV must contain a 'text' column. Columns: {list(df.columns)}")
    df = df[["text"]].copy()

tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
pipe = TextClassificationPipeline(model=model, tokenizer=tok, return_all_scores=False)

preds = pipe(df["text"].tolist())
df["prediction"] = [p["label"] if isinstance(p, dict) else p[0]["label"] for p in preds]

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print("[batch_predict] Wrote", OUT_CSV)
