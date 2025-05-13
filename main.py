# main.py
import joblib
import os
import mlflow
import mlflow.sklearn
import logging
import webbrowser
from steps.ingest import load_and_split_data
from steps.clean import clean_and_transform
from steps.train import train_model
from steps.predict import predict_on_test_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Set MLflow tracking URI
mlflow.set_tracking_uri("file:///C:/ml-project/mlruns")  # Adjust path as needed

def main():
    model = joblib.load('models/model.pkl')
    # Start MLflow experiment
    mlflow.set_experiment("Loan_Default_Prediction")

    with mlflow.start_run() as run:
        # Step 1: Ingest data
        logging.info("Step 1: Ingesting data...")
        load_and_split_data("data/Loan_default.csv", "data/")

        # Step 2: Clean and transform data
        logging.info("Step 2: Cleaning and transforming data...")
        clean_and_transform(
            train_path='data/train.csv',
            test_path='data/test.csv',
            output_dir='data/cleaned',
            preprocessor_path='models/preprocessor.pkl'
        )

        # Step 3: Train the model
        logging.info("Step 3: Training the model...")
        train_model(
            X_train_path="data/cleaned/X_train_clean.csv",
            y_train_path="data/cleaned/y_train.csv",
            X_test_path="data/cleaned/X_test_clean.csv",
            y_test_path="data/cleaned/y_test.csv",
            model_path="models/model.pkl",
            preprocessor_path="models/preprocessor.pkl"
        )

        # Set tags
        mlflow.set_tag("developer", "your_name")
        mlflow.set_tag("model_type", "RandomForestClassifier")
        mlflow.set_tag("pipeline", "Ingestion -> Cleaning -> Training")

        # Step 4 : Predict the model
        result = predict_on_test_data(
        test_path='data/test.csv',
        output_path='data/test_predictions.csv',
        preprocessor_path='models/preprocessor.pkl',
        model_path='models/model.pkl'
        )

        if result:
            accuracy, f1, report, roc = result
            print("Accuracy:", accuracy)
            print("F1 Score:", f1)
            print("Classification Report:\n", report)
            print("ROC AUC Score:", roc)

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc_score", roc)

        # Log model
        mlflow.sklearn.log_model(model, "loan_default_model")

        logging.info("MLflow tracking completed.")
        print("\n=========== EVALUATION RESULTS ===========")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC Score: {roc:.4f}")
        print("\nClassification Report:\n", report)
        print("==========================================")

    # Step 4: Open MLflow UI
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":
    main()
