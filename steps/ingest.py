# steps/ingest.py

import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_and_split_data(input_path, output_dir, test_size=0.2, random_state=42):
    # Load data
    df = pd.read_csv(input_path)
    
    # Split features and target
    X = df.drop(columns=['Default'])  # replace 'default' with your actual target column name
    y = df['Default']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Save split files
    os.makedirs(output_dir, exist_ok=True)
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"Data successfully split and saved to {output_dir}")

if __name__ == "__main__":
    load_and_split_data("data/Loan_default.csv", "data/")
