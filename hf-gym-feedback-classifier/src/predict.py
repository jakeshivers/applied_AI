# src/predict.py
import os, json
from pathlib import Path
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

if not MODEL_DIR.exists():
    raise FileNotFoundError(f"MODEL_DIR not found: {MODEL_DIR}. Train first or set MODEL_DIR.")

# ---------- load pipeline ----------
with open(MODEL_DIR / "labels.json", "r") as f:
    meta = json.load(f)

tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
pipe = TextClassificationPipeline(model=model, tokenizer=tok, return_all_scores=False)

def predict(texts):
    if isinstance(texts, str):
        texts = [texts]
    return pipe(texts)

if __name__ == "__main__":
    samples = [
        "I want to cancel my membership asap before my next billing date",
        "The showers are cold and the lockers are dirty",
        "Please add more yoga classes on weekends",
        "Love the staff and trainers, five stars",
        "I was double charged this month",
    ]
    for s in samples:
        out = predict(s)[0]
        print(s, "->", out)
