# serve_api.py
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from fastapi.responses import HTMLResponse

# --- must match your training architectures ---
class DeepMLP(nn.Module):
    def __init__(self, d_in, d_h1=64, d_h2=32, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_h1), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_h1, d_h2), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_h2, 1)
        )
    def forward(self, x): return self.net(x)

INPUT_DIM = 30  # breast_cancer features; replace if you use another dataset
MODEL_PATH = "artifacts/deep.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# load weights only (safe default in torch>=2.6)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model = DeepMLP(INPUT_DIM)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

# pick a decision threshold (tune on val; hardcode here)
THRESHOLD = 0.50

class Features(BaseModel):
    values: list[float]  # length must equal INPUT_DIM

app = FastAPI(title="Tabular Binary Classifier")

@app.post("/predict")
@torch.no_grad()
def predict(feat: Features):
    import numpy as np
    x = torch.tensor([feat.values], dtype=torch.float32, device=DEVICE)
    logit = model(x)
    prob = torch.sigmoid(logit).item()
    pred = int(prob >= THRESHOLD)
    return {"probability": prob, "prediction": pred, "threshold": THRESHOLD}

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <h3>Tabular Binary Classifier API</h3>
    <p>Use <a href="/docs">/docs</a> to try the <code>POST /predict</code> endpoint.</p>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)




