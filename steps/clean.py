# steps/clean.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import joblib

def clean_and_transform(train_path, test_path, output_dir, preprocessor_path):
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Separate target
    target_column = 'Default'  # Change if your target has a different name
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # Identify column types
    high_cardinality_thresh = 100
    categorical_cols = []
    numerical_cols = []
    for col in X_train.columns:
        unique_vals = X_train[col].nunique()
        if X_train[col].dtype == 'object':
            if unique_vals > high_cardinality_thresh:
                print(f"Dropping high-cardinality column: {col} ({unique_vals} unique values)")
                X_train = X_train.drop(columns=[col])
                X_test = X_test.drop(columns=[col])
            else:
                categorical_cols.append(col)
        elif X_train[col].dtype in ['int64', 'float64']:
            numerical_cols.append(col)
    

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    # Fit on train, transform both
    X_train_clean = pipeline.fit_transform(X_train)
    X_test_clean = pipeline.transform(X_test)

    # Save processed data
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(X_train_clean.toarray() if hasattr(X_train_clean, "toarray") else X_train_clean).to_csv(os.path.join(output_dir, 'X_train_clean.csv'), index=False)
    pd.DataFrame(X_test_clean.toarray() if hasattr(X_test_clean, "toarray") else X_test_clean).to_csv(os.path.join(output_dir, 'X_test_clean.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

    # Save the preprocessor
    joblib.dump(pipeline, preprocessor_path)

    print(f"Cleaned data saved to {output_dir}")
    print(f"Preprocessing pipeline saved to {preprocessor_path}")

if __name__ == "__main__":
    clean_and_transform(
        train_path='data/train.csv',
        test_path='data/test.csv',
        output_dir='data/cleaned',
        preprocessor_path='models/preprocessor.pkl'
    )
