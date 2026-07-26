"""Preprocessing pipeline by Sheren Failla.

Automasi preprocessing dataset Telco Customer Churn (IBM).
Konversi dari notebook eksperimen -> pipeline otomatis yang mengembalikan data siap latih.

Tahapan (sesuai temuan EDA):
1. Drop customerID (identifier, bukan fitur).
2. TotalCharges: object -> numeric; nilai kosong (pelanggan tenure=0) diisi 0.
3. Encoding kolom biner (Yes/No, gender) -> 0/1.
4. One-hot encoding kolom multi-kategori.
5. Split train-test (stratified, target imbalanced ~26.5% churn).
6. Standardisasi fitur numerik kontinu (fit hanya di train -> tanpa data leakage).
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BINARY_MAP = {"Yes": 1, "No": 0, "Female": 1, "Male": 0}
BINARY_COLS = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
MULTI_COLS = ["MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
              "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
              "Contract", "PaymentMethod"]
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def load_data(path):
    """Data loading: baca CSV mentah."""
    return pd.read_csv(path)


def clean_data(df):
    """Drop ID + perbaiki TotalCharges (kosong = pelanggan baru, tenure 0)."""
    df = df.copy()
    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    return df


def encode_features(df):
    """Biner -> 0/1, multi-kategori -> one-hot (drop_first mengurangi kolinearitas)."""
    df = df.copy()
    for col in BINARY_COLS:
        df[col] = df[col].map(BINARY_MAP)
    df = pd.get_dummies(df, columns=MULTI_COLS, drop_first=True, dtype=int)
    return df


def preprocess_data(path, test_size=0.2, random_state=42, save_dir=None):
    """Pipeline lengkap. Mengembalikan X_train, X_test, y_train, y_test siap latih."""
    df = encode_features(clean_data(load_data(path)))

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_test[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        train = X_train.copy(); train["Churn"] = y_train.values
        test = X_test.copy();  test["Churn"] = y_test.values
        train.to_csv(os.path.join(save_dir, "telco_train.csv"), index=False)
        test.to_csv(os.path.join(save_dir, "telco_test.csv"), index=False)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    Xtr, Xte, ytr, yte = preprocess_data("../telco_raw.csv", save_dir="telco_preprocessing")
    print(f"Train: {Xtr.shape} | Test: {Xte.shape} | Churn rate train: {ytr.mean():.3f}")
    print("Preprocessing selesai -> telco_preprocessing/")
