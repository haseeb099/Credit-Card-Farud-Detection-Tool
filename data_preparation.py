import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
from joblib import Parallel, delayed  
import time  


if not os.path.exists("creditcard.csv"):
    print("Error: creditcard.csv not found!")
    exit()


data = pd.read_csv("creditcard.csv")
print("Dataset loaded successfully!")
print("\nFirst 5 rows of the dataset:")
print(data.head())
print("\nClass Distribution (0 = Non-Fraud, 1 = Fraud):")
print(data["Class"].value_counts())


data["Norm_Amount"] = (data["Amount"] - data["Amount"].mean()) / data["Amount"].std()
print("\nFirst 5 rows with Normalized Amount:")
print(data[["Time", "Norm_Amount", "Class"]].head())

X = data.drop(["Class", "Amount"], axis=1)
y = data["Class"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


print("\nTraining set size:", X_train.shape)
print("Test set size:", X_test.shape)
print("Training labels distribution:\n", y_train.value_counts())
print("Test labels distribution:\n", y_test.value_counts())


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


joblib.dump(scaler, "scaler.pkl")
print("Scaler saved as scaler.pkl")


threshold = 0.3


print("\nTraining the Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=50, class_weight="balanced", random_state=42)
rf_model.fit(X_train, y_train)


print("\nTraining the Logistic Regression model...")
lr_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)


print("\nTraining the XGBoost model...")
xgb_model = XGBClassifier(scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]), 
                          n_estimators=50, random_state=42)
xgb_model.fit(X_train, y_train)


def predict_with_timing(model, X, model_name):
    start_time = time.time()
    prob = model.predict_proba(X)[:, 1]
    elapsed = time.time() - start_time
    print(f"{model_name} prediction time: {elapsed:.4f} seconds")
    return prob


def parallel_predictions(models, X_unscaled, X_scaled):
    results = Parallel(n_jobs=3)(  # Use 3 parallel jobs
        delayed(predict_with_timing)(model, X, name)
        for model, X, name in [
            (rf_model, X_unscaled, "Random Forest"),
            (lr_model, X_scaled, "Logistic Regression"),
            (xgb_model, X_unscaled, "XGBoost")
        ]
    )
    return results


print("\nMaking predictions...")


rf_prob, lr_prob, xgb_prob = parallel_predictions(
    [rf_model, lr_model, xgb_model], X_test, X_test_scaled
)


rf_pred_adjusted = (rf_prob >= threshold).astype(int)
lr_pred_adjusted = (lr_prob >= threshold).astype(int)
xgb_pred_adjusted = (xgb_prob >= threshold).astype(int)


ensemble_prob = np.mean([rf_prob, lr_prob, xgb_prob], axis=0)
ensemble_pred_adjusted = (ensemble_prob >= threshold).astype(int)


print(f"\nRandom Forest Performance (threshold = {threshold}):")
print(classification_report(y_test, rf_pred_adjusted))

print(f"\nLogistic Regression Performance (threshold = {threshold}):")
print(classification_report(y_test, lr_pred_adjusted))

print(f"\nXGBoost Performance (threshold = {threshold}):")
print(classification_report(y_test, xgb_pred_adjusted))


print(f"\nEnsemble (Averaged Probabilities) Performance (threshold = {threshold}):")
print(classification_report(y_test, ensemble_pred_adjusted))


print("\nSaving models...")
joblib.dump(rf_model, "fraud_model_rf.pkl")
joblib.dump(lr_model, "fraud_model_lr.pkl")
joblib.dump(xgb_model, "fraud_model_xgb.pkl")
print("Models saved as fraud_model_rf.pkl, fraud_model_lr.pkl, and fraud_model_xgb.pkl")


print("\nSetting up monitoring framework...")
log_file = "fraud_detection_log.csv"
if not os.path.exists(log_file):
    pd.DataFrame(columns=["timestamp", "transaction_id", "rf_prob", "lr_prob", "xgb_prob", 
                          "ensemble_prob", "prediction", "true_label"]).to_csv(log_file, index=False)


def predict_and_log(transaction, transaction_id, true_label=None):
    
    norm_amount = (transaction[-1] - data["Amount"].mean()) / data["Amount"].std()
    transaction[-1] = norm_amount
    X_unscaled = np.array([transaction])
    X_scaled = scaler.transform(X_unscaled)
    
    
    rf_p, lr_p, xgb_p = parallel_predictions([rf_model, lr_model, xgb_model], X_unscaled, X_scaled)
    ens_p = np.mean([rf_p, lr_p, xgb_p])
    pred = 1 if ens_p >= threshold else 0
    
    
    log_entry = pd.DataFrame({
        "timestamp": [pd.Timestamp.now()], "transaction_id": [transaction_id],
        "rf_prob": [rf_p[0]], "lr_prob": [lr_p[0]], "xgb_prob": [xgb_p[0]],
        "ensemble_prob": [ens_p], "prediction": [pred], "true_label": [true_label]
    })
    log_entry.to_csv(log_file, mode="a", header=False, index=False)
    return pred, ens_p


dummy_transaction = X_test.iloc[0].values  
pred, prob = predict_and_log(dummy_transaction, "test_001", y_test.iloc[0])
print(f"\nReal-time prediction for test_001: Probability = {prob:.3f}, Prediction = {pred}")


print("\nNote: Monitor {log_file} and retrain models periodically with new data when performance degrades.")

# Save X_test
#X_test.to_csv("X_test.csv", index=False)
#print("Test data saved as X_test.csv!")

X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)  
print("Test data saved as X_test.csv and y_test.csv!")