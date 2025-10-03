
# Gym Feedback Classifier (Hugging Face mini-project)

Classify incoming gym/member feedback into categories:
- `cancel_intent`
- `billing_issue`
- `facility_complaint`
- `positive_feedback`
- `class_request`

This project fine-tunes **DistilBERT** with Hugging Face **Transformers** and **Trainer** on a small synthetic dataset so you can run end-to-end today.

## Quickstart

```bash
# 0) Create venv (recommended)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 1) Install deps
pip install -r requirements.txt

# 2) Split dataset (produces data/train.csv and data/valid.csv)
python scripts/split_dataset.py --input_csv data/gym_feedback.csv --train_csv data/train.csv --valid_csv data/valid.csv --valid_frac 0.2

# 3) Train
python src/train.py

# 4) Predict
python src/predict.py
```

Artifacts are saved in `artifacts/model` (config.json, tokenizer.json, pytorch_model.bin, labels.json).

## Notes
- Synthetic data is included for reproducibility. Replace `data/gym_feedback.csv` with your real feedback export (`text,label` columns) to adapt.
- The Trainer runs for 3 epochs; tweak hyperparameters via env vars or in `src/train.py`.
- Evaluation metrics: accuracy and macro-F1 (good when class sizes differ).

## Requirements
See `requirements.txt`.
