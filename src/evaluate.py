import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import json

from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==============================
# PATHS
# ==============================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "disease_data.csv"

MODEL_PATH = BASE_DIR / "models" / "disease_prediction_model.pkl"

PLOTS_DIR = BASE_DIR / "data" / "processed" / "plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["location", "disease", "date"]
).reset_index(drop=True)


# ==============================
# CREATE TARGET
# ==============================

df["target_cases"] = (
    df.groupby(
        ["location", "disease"]
    )["cases"].shift(-1)
)

df = df.dropna(
    subset=["target_cases"]
).copy()


# ==============================
# FEATURE ENGINEERING
# ==============================

df["month"] = df["date"].dt.month

df["week"] = (
    df["date"]
    .dt.isocalendar()
    .week
    .astype(int)
)


# ==============================
# ONE-HOT ENCODING
# ==============================

df = pd.get_dummies(
    df,
    columns=["location", "disease"],
    dtype=int
)


# ==============================
# LOAD TRAINED MODEL
# ==============================

model_data = joblib.load(MODEL_PATH)

model = model_data["model"]

features = model_data["features"]


# ==============================
# PREPARE FEATURES
# ==============================

X = df.drop(
    columns=["target_cases", "date"],
    errors="ignore"
)

X = X.reindex(
    columns=features,
    fill_value=0
)

y = df["target_cases"]


# ==============================
# CHRONOLOGICAL TEST SPLIT
# ==============================

split_index = int(len(X) * 0.8)

X_test = X.iloc[split_index:]

y_test = y.iloc[split_index:]


# ==============================
# MODEL PREDICTIONS
# ==============================

predictions = model.predict(X_test)

predictions = np.maximum(
    predictions,
    0
)


# ==============================
# BASELINE PREDICTIONS
# ==============================

# Baseline assumption:
# next period cases = current cases

baseline_predictions = X_test["cases"].values

baseline_predictions = np.maximum(
    baseline_predictions,
    0
)


# ==============================
# MODEL METRICS
# ==============================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


# ==============================
# BASELINE METRICS
# ==============================

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_predictions
    )
)

baseline_r2 = r2_score(
    y_test,
    baseline_predictions
)


# ==============================
# PRINT RESULTS
# ==============================

print("\n================================")
print("MODEL EVALUATION")
print("================================")

print("\nRandom Forest Model")

print(f"MAE  : {mae:.2f}")

print(f"RMSE : {rmse:.2f}")

print(f"R²   : {r2:.4f}")


print("\nBaseline Model")
print("(Next period cases = current cases)")

print(f"MAE  : {baseline_mae:.2f}")

print(f"RMSE : {baseline_rmse:.2f}")

print(f"R²   : {baseline_r2:.4f}")

print("\n================================")


# ==============================
# FEATURE IMPORTANCE
# ==============================

importance = model.feature_importances_

feature_importance = pd.DataFrame({

    "feature": features,

    "importance": importance

})


feature_importance = (
    feature_importance
    .sort_values(
        "importance",
        ascending=False
    )
)


print("\nTop Important Features:")

print(
    feature_importance.head(10)
)


# ==============================
# SAVE EVALUATION REPORT
# ==============================

report_path = (
    BASE_DIR /
    "data" /
    "processed" /
    "model_evaluation.txt"
)


with open(report_path, "w") as file:

    file.write(
        "AI Disease Outbreak Prediction System\n"
    )

    file.write(
        "Model Evaluation Report\n"
    )

    file.write(
        "================================\n\n"
    )

    file.write(
        "Model: Random Forest Regression\n"
    )

    file.write(
        f"Test Samples: {len(X_test)}\n\n"
    )

    file.write(
        "Random Forest Metrics\n"
    )

    file.write(
        f"MAE: {mae:.4f}\n"
    )

    file.write(
        f"RMSE: {rmse:.4f}\n"
    )

    file.write(
        f"R2 Score: {r2:.4f}\n\n"
    )

    file.write(
        "Baseline Metrics\n"
    )

    file.write(
        "Baseline: Next period cases = current cases\n"
    )

    file.write(
        f"MAE: {baseline_mae:.4f}\n"
    )

    file.write(
        f"RMSE: {baseline_rmse:.4f}\n"
    )

    file.write(
        f"R2 Score: {baseline_r2:.4f}\n"
    )


# ==============================
# SAVE EVALUATION JSON
# ==============================

evaluation_data = {

    "model": "Random Forest Regression",

    "test_samples":
        int(len(X_test)),

    "metrics": {

        "mae":
            round(float(mae), 4),

        "rmse":
            round(float(rmse), 4),

        "r2":
            round(float(r2), 4)

    },

    "baseline": {

        "description":
            "Next period cases = current cases",

        "mae":
            round(
                float(baseline_mae),
                4
            ),

        "rmse":
            round(
                float(baseline_rmse),
                4
            ),

        "r2":
            round(
                float(baseline_r2),
                4
            )

    },

    "feature_importance": [

        {

            "feature":
                str(row["feature"]),

            "importance":
                round(
                    float(row["importance"]),
                    6
                )

        }

        for _, row
        in feature_importance.head(10).iterrows()

    ]

}


evaluation_json_path = (
    BASE_DIR /
    "data" /
    "processed" /
    "model_evaluation.json"
)


with open(
    evaluation_json_path,
    "w"
) as file:

    json.dump(
        evaluation_data,
        file,
        indent=4
    )


# ==============================
# ACTUAL VS PREDICTED GRAPH
# ==============================

plt.figure(figsize=(12, 6))

plt.plot(
    y_test.values,
    label="Actual Cases"
)

plt.plot(
    predictions,
    label="Random Forest Predictions"
)

plt.plot(
    baseline_predictions,
    label="Baseline Predictions"
)

plt.title(
    "Actual vs Predicted Disease Cases"
)

plt.xlabel(
    "Test Sample"
)

plt.ylabel(
    "Cases"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    PLOTS_DIR /
    "actual_vs_predicted.png",
    dpi=150
)

plt.close()


# ==============================
# FEATURE IMPORTANCE GRAPH
# ==============================

top_features = (
    feature_importance
    .head(10)
    .sort_values(
        "importance"
    )
)


plt.figure(figsize=(10, 6))

plt.barh(
    top_features["feature"],
    top_features["importance"]
)

plt.title(
    "Top 10 Feature Importances"
)

plt.xlabel(
    "Importance"
)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR /
    "feature_importance.png",
    dpi=150
)

plt.close()


# ==============================
# COMPLETION MESSAGE
# ==============================

print("\nEvaluation completed successfully.")

print(
    f"\nReport saved to:\n{report_path}"
)

print(
    f"\nJSON report saved to:\n"
    f"{evaluation_json_path}"
)

print(
    "\nGraphs saved to:\n"
    f"{PLOTS_DIR}"
)