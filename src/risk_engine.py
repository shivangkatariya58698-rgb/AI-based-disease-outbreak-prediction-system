import pandas as pd


DATA_PATH = "data/raw/disease_data.csv"


def load_historical_data():

    df = pd.read_csv(DATA_PATH)

    return df


def calculate_risk(
    current_cases,
    predicted_cases,
    historical_cases
):

    # Avoid division by zero
    if current_cases <= 0:
        growth_rate = 0
    else:
        growth_rate = (
            (predicted_cases - current_cases)
            / current_cases
        ) * 100

    # Historical statistics
    historical_mean = historical_cases.mean()
    historical_std = historical_cases.std()

    # Determine case level
    if predicted_cases >= (
        historical_mean + 2 * historical_std
    ):
        case_level = "High"

    elif predicted_cases >= (
        historical_mean + historical_std
    ):
        case_level = "Moderate"

    else:
        case_level = "Low"

    # Determine growth level
    if growth_rate >= 50:
        growth_level = "High"

    elif growth_rate >= 20:
        growth_level = "Moderate"

    else:
        growth_level = "Low"

    # Convert levels to scores
    level_scores = {
        "Low": 1,
        "Moderate": 2,
        "High": 3
    }

    case_score = level_scores[case_level]
    growth_score = level_scores[growth_level]

    # Combined risk score
    risk_score = (
        case_score * 0.6
        + growth_score * 0.4
    )

    # Final risk classification
    if risk_score >= 2.5:
        risk = "High"

    elif risk_score >= 1.5:
        risk = "Moderate"

    else:
        risk = "Low"

    return {
        "predicted_cases": round(
            predicted_cases,
            2
        ),
        "growth_rate": round(
            growth_rate,
            2
        ),
        "historical_mean": round(
            historical_mean,
            2
        ),
        "case_level": case_level,
        "growth_level": growth_level,
        "risk_score": round(
            risk_score,
            2
        ),
        "risk": risk
    }


def main():

    df = load_historical_data()

    # Example
    location = "Dehradun"
    disease = "Dengue"

    filtered = df[
        (df["location"] == location)
        &
        (df["disease"] == disease)
    ]

    current_cases = (
        filtered["cases"]
        .iloc[-1]
    )

    historical_cases = (
        filtered["cases"]
    )

    # Example prediction
    predicted_cases = current_cases * 1.5

    result = calculate_risk(
        current_cases,
        predicted_cases,
        historical_cases
    )

    print("\n========== OUTBREAK RISK ==========")

    for key, value in result.items():
        print(
            f"{key}: {value}"
        )


if __name__ == "__main__":
    main()