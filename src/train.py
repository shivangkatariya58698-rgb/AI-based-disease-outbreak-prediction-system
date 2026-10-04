import os

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


DATA_PATH = "data/raw/disease_data.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "disease_prediction_model.pkl"
)


def load_data():

    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"])

    # Sort chronologically
    df = df.sort_values(
        ["location", "disease", "date"]
    )

    # Target = next week's cases
    df["target_cases"] = (
        df.groupby(
            ["location", "disease"]
        )["cases"].shift(-1)
    )

    # Remove rows without target
    df = df.dropna(
        subset=["target_cases"]
    )

    return df


def prepare_features(df):

    # Convert categorical variables to numerical values
    df = pd.get_dummies(
        df,
        columns=["location", "disease"],
        dtype=int
    )

    # Date-based features
    df["month"] = df["date"].dt.month
    df["week"] = df["date"].dt.isocalendar().week.astype(int)

    feature_columns = [
        "cases",
        "temperature",
        "humidity",
        "rainfall",
        "population",
        "previous_cases",
        "month",
        "week"
    ]

    # Add encoded location and disease columns
    encoded_columns = [
        column
        for column in df.columns
        if column.startswith("location_")
        or column.startswith("disease_")
    ]

    feature_columns.extend(encoded_columns)

    X = df[feature_columns]

    y = df["target_cases"]

    return X, y, feature_columns


def train_model(X_train, y_train):

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def main():

    print("Loading dataset...")

    df = load_data()

    print(f"Total records: {len(df)}")

    X, y, feature_columns = prepare_features(df)

    # Chronological split
    split_index = int(len(X) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    print("\nTraining Random Forest model...")

    model = train_model(
        X_train,
        y_train
    )

    print("Model training completed.")

    # Predictions
    predictions = model.predict(X_test)

    # Evaluation
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    print("\n========== MODEL PERFORMANCE ==========")

    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    # Create model directory
    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # Save model + feature information
    model_data = {
        "model": model,
        "features": feature_columns
    }

    joblib.dump(
        model_data,
        MODEL_PATH
    )

    print("\nModel saved successfully:")
    print(MODEL_PATH)


if __name__ == "__main__":
    main()