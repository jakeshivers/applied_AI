# Neural Network Demo

This project shows how to train a simple neural network on a real dataset, evaluate its performance, and then serve it through an API.

The goal is to make machine learning feel less abstract: you train a model on data, save what it learns, and then expose it so anyone can send in new examples and get predictions back.

# What’s Inside

### 1. Training Script (train_nn_tabular.py)
- Loads a sample dataset (Breast Cancer data from scikit-learn).
- Trains three models:
    - Logistic Regression (a simple baseline)
    - A small neural net with one hidden layer
    - A deeper neural net with two hidden layers
- Compares results using accuracy, ROC curves, and precision-recall curves.
- Saves the best models into an artifacts/ folder.

### 2. API Service (serve_api.py)
- Loads a trained neural network from artifacts/.
- Exposes a /predict endpoint where you can send in new data as JSON.
- Returns a probability and a prediction (0 or 1).

### 3. Artifacts
- Saved model weights.
- Evaluation plots (ROC and PR curves).


# Why This Matters

- Shows progression: you start with a simple model, then add hidden layers to show how neural networks learn more complex patterns.

- Demonstrates end-to-end ML: not just training, but also deployment.

- Hands-on: you can actually run the API and send requests to see predictions in real time.

# How to Run It

### 1. Train a model:
```bash
python neural-network/train_nn_tabular.py
```
This saves models and plots to `artifacts/`.

### 2. Start the API
```bash
python neural-network/serve_api.py
```
The service will run on `http://localhost:8000`.

### 3. Try a prdiction
Open `http://localhost:8000/docs` in your browser.
Use the built-in interface to send a test request and see the response.

# Example Prediction
Send this request:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"values":[0.12,-1.03,0.44,0.08,-0.55,0.31,0.02,-0.14,0.66,-0.27,0.04,0.19,-0.41,0.23,0.37,-0.11,0.52,0.07,-0.33,0.29,0.05,-0.22,0.18,0.12,-0.07,0.26,-0.15,0.09,0.01,0.17]}'
```

Response:
```python
{"probability": 0.62, "prediction": 1, "threshold": 0.5}
```

# Key Takeaways
- Neural networks are just layers of math that get tuned on data.
- Once trained, you can wrap them in an API to make predictions available to anyone.
- This project is a small but complete example of that process.