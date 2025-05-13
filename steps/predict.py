from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
import pandas as pd
import joblib
import os

def predict_on_test_data(
    test_path,
    output_path,
    preprocessor_path,
    model_path
):
    df = pd.read_csv(test_path)

    has_labels = 'Default' in df.columns
    if has_labels:
        y_true = df['Default']
        df_features = df.drop(columns=['Default'])
    else:
        y_true = None
        df_features = df

    preprocessor = joblib.load(preprocessor_path)
    X = preprocessor.transform(df_features)

    model = joblib.load(model_path)

    predictions = model.predict(X)
    prediction_probs = model.predict_proba(X)[:, 1]

    df['Predicted_Default'] = predictions
    df['Default_Probability'] = prediction_probs

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Predictions on test data saved to {output_path}")

    if has_labels:
        accuracy = accuracy_score(y_true, predictions)
        f1 = f1_score(y_true, predictions)
        report = classification_report(y_true, predictions)
        roc = roc_auc_score(y_true, prediction_probs)
        return accuracy, f1, report, roc
    else:
        return None
