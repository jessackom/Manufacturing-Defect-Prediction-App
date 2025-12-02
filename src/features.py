import pandas as pd
from sklearn.preprocessing import StandardScaler

def basic_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply minimal feature engineering: drop NA, convert dtypes as needed."""
    df = df.copy()
    df = df.dropna()
    # Placeholder: convert categorical columns, generate aggregates, etc.
    return df

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
