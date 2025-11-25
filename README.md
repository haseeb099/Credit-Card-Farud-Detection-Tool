🚀 Machine Learning + Real-Time Simulation GUI

This project implements a real-time credit card fraud detection system using multiple machine-learning models (Random Forest, Logistic Regression, and XGBoost) combined into an ensemble model. It also includes a Tkinter GUI application that simulates live transactions, generates PCA-like features, and predicts fraud probability in real time.

📂 Project Structure
│── data_preparation.py          # Train models, preprocess data, save test set & logs
│── real_time_simulation_gui.py  # Real-time prediction GUI app
│── fraud_model_rf.pkl           # Random Forest model
│── fraud_model_lr.pkl           # Logistic Regression model
│── fraud_model_xgb.pkl          # XGBoost model
│── scaler.pkl                   # StandardScaler object
│── fraud_detection_log.csv      # Auto-generated logging file
│── X_test.csv                   # Saved test data
│── y_test.csv                   # Saved test labels

✨ Features
🔍 1. Data Preparation & Model Training

Loads and processes the Kaggle Credit Card Fraud Dataset

Normalizes transaction amounts

Splits data using stratified sampling

Trains:

🌲 Random Forest

📉 Logistic Regression

⚡ XGBoost

Saves:

Trained models (.pkl)

Scaler

Test data for simulation

Creates a parallel prediction system for speed optimization

Builds an ensemble predictor using averaged model probabilities

🖥️ 2. Real-Time Fraud Detection GUI

Built using Tkinter, the GUI allows real-time simulations:

User Inputs:

Transaction ID

Amount (€)

Time (HH:MM)

Merchant category (Grocery, Online, Travel, Other)

Location (Local / Foreign)

Under the hood:

Generates synthetic PCA-style V1–V28 features

Applies all 3 ML models in parallel

Shows:

Ensemble probability

Individual model probabilities

Model prediction times

Logs each prediction into fraud_detection_log.csv

🧠 Ensemble Logic

The GUI averages predictions from:

Random Forest

Logistic Regression

XGBoost

ensemble_prob = np.mean([rf_prob, lr_prob, xgb_prob])
prediction = 1 if ensemble_prob >= threshold else 0


Threshold can be fine-tuned (default: 0.30).

📊 Logging System

Every prediction is automatically stored with:

Timestamp

Transaction ID

RF / LR / XGB probabilities

Ensemble probability

Final prediction

True label (if available)

This supports real-time monitoring and model drift detection.

▶️ How to Run
1. Train models

Make sure your dataset is named:

creditcard.csv


Then run:

python data_preparation.py


This will generate all model files and logs.

2. Start the GUI
python real_time_simulation_gui.py


The GUI will open and allow you to simulate transactions.

📈 Future Improvements

You can extend this project with:

API endpoint using FastAPI / Flask

Dashboard using Streamlit

Model drift detection

Auto-retraining pipeline

Anomaly detection algorithms

Database integration (MongoDB / PostgreSQL)

📜 License

This project is open-source and free to use.
