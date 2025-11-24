import pandas as pd
import numpy as np
import joblib
import time
import os
from datetime import datetime
from joblib import Parallel, delayed
import tkinter as tk
from tkinter import ttk, messagebox


try:
    rf_model = joblib.load("fraud_model_rf.pkl")
    lr_model = joblib.load("fraud_model_lr.pkl")
    xgb_model = joblib.load("fraud_model_xgb.pkl")
    scaler = joblib.load("scaler.pkl")
    print("Models and scaler loaded successfully!")
except FileNotFoundError as e:
    print(f"Error: Model file not found - {e}")
    exit(1)
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)


def predict_with_timing(model, X, model_name):
    start_time = time.time()
    prob = model.predict_proba(X)[:, 1]
    elapsed = time.time() - start_time
    return prob, elapsed


def parallel_predictions(X_unscaled, X_scaled):
    results = Parallel(n_jobs=3)(
        delayed(predict_with_timing)(model, X, name)
        for model, X, name in [
            (rf_model, X_unscaled, "Random Forest"),
            (lr_model, X_scaled, "Logistic Regression"),
            (xgb_model, X_unscaled, "XGBoost")
        ]
    )
    rf_prob, rf_time = results[0]
    lr_prob, lr_time = results[1]
    xgb_prob, xgb_time = results[2]
    return rf_prob[0], lr_prob[0], xgb_prob[0], rf_time, lr_time, xgb_time

def simulate_pca_features(amount, time_seconds, merchant_category, location):
    
    v_values = np.random.normal(0, 0.5, 28)
    fraud_score = 0

    
    if amount < 100 and merchant_category == "Grocery" and location == "Local":
        v_values = np.random.normal(0, 0.3, 28)  # Tighter range for legit
        v_values = np.clip(v_values, -1.5, 1.5)  # Tame values
        print(f"Simulated V1-V28 (Legit): {v_values[:5]}... (first 5 shown)")
        return v_values.tolist()

    #
    if amount > 500:  
        fraud_score += 3
        v_values[0:5] = np.random.uniform(-15, 15, 5)  
    if merchant_category in ["Online", "Travel"]:
        fraud_score += 1
        v_values[5:8] = np.random.uniform(-5, 5, 3)  
    if location == "Foreign":
        fraud_score += 2
        v_values[10:13] = np.random.uniform(-10, 10, 3)  
    
    
    if fraud_score >= 4:
        v_values[14:17] = np.random.uniform(-20, 20, 3)  
    
    print(f"Simulated V1-V28 (Fraud score {fraud_score}): {v_values[:5]}... (first 5 shown)")
    return v_values.tolist()


def predict_and_log(transaction, transaction_id, threshold=0.34):
    X_unscaled = pd.DataFrame([transaction], columns=feature_names)
    X_scaled = scaler.transform(X_unscaled)
    print(f"Scaled input for LR (first 5): {X_scaled[0][:5]}")  
    
    rf_prob, lr_prob, xgb_prob, rf_time, lr_time, xgb_time = parallel_predictions(X_unscaled, X_scaled)
    ensemble_prob = np.mean([rf_prob, lr_prob, xgb_prob])
    prediction = 1 if ensemble_prob >= threshold else 0
    
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "transaction_id": transaction_id,
        "rf_prob": rf_prob,
        "lr_prob": lr_prob,
        "xgb_prob": xgb_prob,
        "ensemble_prob": ensemble_prob,
        "prediction": prediction,
        "true_label": "N/A"
    }
    log_df = pd.DataFrame([log_entry])
    log_df.to_csv(log_file, mode="a", header=not os.path.exists(log_file) or os.stat(log_file).st_size == 0, index=False)
    
    return prediction, ensemble_prob, rf_prob, lr_prob, xgb_prob, rf_time, lr_time, xgb_time

feature_names = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 
                 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 
                 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Norm_Amount']


log_file = "fraud_detection_log.csv"
if not os.path.exists(log_file):
    pd.DataFrame(columns=["timestamp", "transaction_id", "rf_prob", "lr_prob", "xgb_prob", 
                          "ensemble_prob", "prediction", "true_label"]).to_csv(log_file, index=False)
print(f"Logging to {log_file}")


class FraudDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Fraud Detection")
        self.root.geometry("800x400")

        ttk.Label(root, text="Transaction ID:").grid(row=0, column=0, padx=5, pady=5)
        self.transaction_id_entry = ttk.Entry(root)
        self.transaction_id_entry.grid(row=0, column=1, padx=5, pady=5)
        self.transaction_id_entry.insert(0, "test_001")

        ttk.Label(root, text="Transaction Amount (Є):").grid(row=1, column=0, padx=5, pady=5)
        self.amount_entry = ttk.Entry(root)
        self.amount_entry.grid(row=1, column=1, padx=5, pady=5)
        self.amount_entry.insert(0, "50.25")

        ttk.Label(root, text="Time (HH:MM):").grid(row=2, column=0, padx=5, pady=5)
        self.time_entry = ttk.Entry(root)
        self.time_entry.grid(row=2, column=1, padx=5, pady=5)
        self.time_entry.insert(0, "14:30")

        ttk.Label(root, text="Merchant Category:").grid(row=3, column=0, padx=5, pady=5)
        self.merchant_combo = ttk.Combobox(root, values=["Grocery", "Online", "Travel", "Other"])
        self.merchant_combo.grid(row=3, column=1, padx=5, pady=5)
        self.merchant_combo.set("Grocery")

        ttk.Label(root, text="Location:").grid(row=4, column=0, padx=5, pady=5)
        self.location_combo = ttk.Combobox(root, values=["Local", "Foreign"])
        self.location_combo.grid(row=4, column=1, padx=5, pady=5)
        self.location_combo.set("Local")

        self.predict_button = ttk.Button(root, text="Predict", command=self.predict)
        self.predict_button.grid(row=5, column=0, columnspan=2, pady=10)

        self.result_text = tk.Text(root, height=10, width=80)
        self.result_text.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

    def predict(self):
        try:
            transaction_id = self.transaction_id_entry.get()
            amount = float(self.amount_entry.get())
            time_str = self.time_entry.get()
            merchant_category = self.merchant_combo.get()
            location = self.location_combo.get()

            hours, minutes = map(int, time_str.split(":"))
            time_seconds = hours * 3600 + minutes * 60

            v_values = simulate_pca_features(amount, time_seconds, merchant_category, location)
            transaction = [time_seconds] + v_values + [amount]
            
            pred, ens_prob, rf_prob, lr_prob, xgb_prob, rf_time, lr_time, xgb_time = predict_and_log(transaction, transaction_id)
            
            result = f"Transaction ID: {transaction_id}\n"
            result += f"Prediction: {'Fraudulent' if pred == 1 else 'Legitimate'}\n"
            result += f"Ensemble Probability: {ens_prob:.3f}\n"
            result += f"Random Forest Probability: {rf_prob:.3f} (Time: {rf_time:.4f}s)\n"
            result += f"Logistic Regression Probability: {lr_prob:.3f} (Time: {lr_time:.4f}s)\n"
            result += f"XGBoost Probability: {xgb_prob:.3f} (Time: {xgb_time:.4f}s)\n"
            result += f"Threshold Used: 0.3\n"
            result += f"Result logged to {log_file}"

            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, result)
            self.result_text.tag_configure("fraud", foreground="red")
            self.result_text.tag_configure("legit", foreground="green")
            self.result_text.tag_add("fraud" if pred == 1 else "legit", "2.0", "2.end")

        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid values (e.g., numeric amount, HH:MM time).")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FraudDetectionApp(root)
    root.mainloop()