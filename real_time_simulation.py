import pandas as pd
import numpy as np
import joblib
import time
import os
from datetime import datetime
from joblib import Parallel, delayed
from sklearn.metrics import classification_report


rf_model = joblib.load("fraud_model_rf.pkl")
lr_model = joblib.load("fraud_model_lr.pkl")
xgb_model = joblib.load("fraud_model_xgb.pkl")
scaler = joblib.load("scaler.pkl")
print("Models and scaler loaded successfully!")


X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv") if os.path.exists("y_test.csv") else None  # Optional: for evaluation
print("Loaded test data for simulation:", X_test.shape)


def predict_with_timing(model, X, model_name):
    start_time = time.time()
    prob = model.predict_proba(X)[:, 1]
    elapsed = time.time() - start_time
    print(f"{model_name} prediction time: {elapsed:.4f} seconds")
    return prob


def parallel_predictions(models, X_unscaled, X_scaled):
    results = Parallel(n_jobs=3)(
        delayed(predict_with_timing)(model, X, name)
        for model, X, name in [
            (rf_model, X_unscaled, "Random Forest"),
            (lr_model, X_scaled, "Logistic Regression"),
            (xgb_model, X_unscaled, "XGBoost")
        ]
    )
    return results[0], results[1], results[2]  

def predict_and_log(transaction, transaction_id, true_label=None, threshold=0.3):
    
    X_unscaled = pd.DataFrame([transaction], columns=X_test.columns)
    X_scaled = scaler.transform(X_unscaled)
    
    
    rf_prob, lr_prob, xgb_prob = parallel_predictions([rf_model, lr_model, xgb_model], X_unscaled, X_scaled)
    ensemble_prob = np.mean([rf_prob, lr_prob, xgb_prob])
    prediction = 1 if ensemble_prob >= threshold else 0
    
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "transaction_id": transaction_id,
        "rf_prob": rf_prob[0],
        "lr_prob": lr_prob[0],
        "xgb_prob": xgb_prob[0],
        "ensemble_prob": ensemble_prob,
        "prediction": prediction,
        "true_label": true_label if true_label is not None else "N/A"
    }
    log_df = pd.DataFrame([log_entry])
    log_df.to_csv(log_file, mode="a", header=not os.path.exists(log_file) or os.stat(log_file).st_size == 0, index=False)
    
    return prediction, ensemble_prob


log_file = "fraud_detection_log.csv"
if not os.path.exists(log_file):
    pd.DataFrame(columns=["timestamp", "transaction_id", "rf_prob", "lr_prob", "xgb_prob", 
                          "ensemble_prob", "prediction", "true_label"]).to_csv(log_file, index=False)
print(f"Logging to {log_file}")


print("\nStarting real-time fraud detection simulation...")
num_transactions = 10  
predictions = []
true_labels = []

for i in range(num_transactions):
    
    transaction = X_test.iloc[i].values
    true_label = y_test.iloc[i].values[0] if y_test is not None else None
    transaction_id = f"test_{i:03d}"
    
    
    pred, prob = predict_and_log(transaction, transaction_id, true_label)
    predictions.append(pred)
    if true_label is not None:
        true_labels.append(true_label)
    
    
    print(f"Transaction {transaction_id}: Probability = {prob:.3f}, Prediction = {pred}")
    
    
    time.sleep(1)  


if y_test is not None and len(true_labels) > 0:
    print("\nSimulation Performance (Ensemble, threshold = 0.3):")
    print(classification_report(true_labels, predictions))


print(f"\nSimulation complete. Check {log_file} for detailed results.")
print("Note: Monitor log file and retrain models periodically if performance degrades.")