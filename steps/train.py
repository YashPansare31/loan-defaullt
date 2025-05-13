# steps/train.py

import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report
import mlflow
import mlflow.sklearn

def train_model(
    X_train_path,
    y_train_path,
    X_test_path,
    y_test_path,
    model_path,
    preprocessor_path
):
    # Load data
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()


    selected_features = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'HasMortgage', 'HasDependents']
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    # Load preprocessor (if needed later in app/predict)
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")

    # Start MLflow experiment

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    # Predict
    y_pred = model.predict(X_test)
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    # Log to MLflow
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(model, "random_forest_model")
    # Save model locally
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model trained and saved to {model_path}")
    print(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    print(f"Classification Report:{report}")
          
if __name__ == "__main__":
    train_model(
        X_train_path="data/cleaned/X_train_clean.csv",
        y_train_path="data/cleaned/y_train.csv",
        X_test_path="data/cleaned/X_test_clean.csv",
        y_test_path="data/cleaned/y_test.csv",
        model_path="models/model.pkl",
        preprocessor_path="models/preprocessor.pkl"
    )
