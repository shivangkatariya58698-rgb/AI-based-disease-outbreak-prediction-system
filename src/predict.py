import joblib
import pandas as pd
from pathlib import Path


MODEL_PATH = Path("models/disease_prediction_model.pkl")


def predict_cases(
    cases,
    temperature,
    humidity,
    rainfall,
    population,
    previous_cases,
    location,
    disease,
    forecast_date
):
    """
    Predict disease cases for a single forecast date.
    """

    model_data = joblib.load(MODEL_PATH)

    model = model_data["model"]
    feature_columns = model_data["features"]

    forecast_date = pd.to_datetime(forecast_date)

    month = forecast_date.month
    week = int(forecast_date.isocalendar().week)

    input_data = pd.DataFrame({
        "cases": [cases],
        "temperature": [temperature],
        "humidity": [humidity],
        "rainfall": [rainfall],
        "population": [population],
        "previous_cases": [previous_cases],
        "month": [month],
        "week": [week]
    })

    # Add location one-hot encoded columns
    for column in feature_columns:
        if column.startswith("location_"):
            input_data[column] = (
                1 if column == f"location_{location}" else 0
            )

    # Add disease one-hot encoded columns
    for column in feature_columns:
        if column.startswith("disease_"):
            input_data[column] = (
                1 if column == f"disease_{disease}" else 0
            )

    # Make sure feature order exactly matches training
    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    prediction = model.predict(input_data)[0]

    # Cases cannot be negative
    prediction = max(0, float(prediction))

    return prediction


def predict_7_days(
    cases,
    temperature,
    humidity,
    rainfall,
    population,
    previous_cases,
    location,
    disease,
    start_date
):
    """
    Generate predictions for the next 7 days.

    The first prediction uses the user's current inputs.
    Each following day uses the previous prediction as
    the current case value.
    """

    start_date = pd.to_datetime(start_date)

    forecasts = []

    current_cases = float(cases)
    previous_week_cases = float(previous_cases)

    for day in range(1, 8):

        forecast_date = start_date + pd.Timedelta(days=day)

        predicted_cases = predict_cases(
            cases=current_cases,
            temperature=temperature,
            humidity=humidity,
            rainfall=rainfall,
            population=population,
            previous_cases=previous_week_cases,
            location=location,
            disease=disease,
            forecast_date=forecast_date
        )

        forecasts.append({
            "day": day,
            "date": forecast_date.strftime("%Y-%m-%d"),
            "predicted_cases": round(predicted_cases, 2)
        })

        # Feed prediction forward into the next day
        previous_week_cases = current_cases
        current_cases = predicted_cases

    return forecasts


if __name__ == "__main__":

    print("=" * 50)
    print("7-DAY DISEASE OUTBREAK FORECAST")
    print("=" * 50)

    forecasts = predict_7_days(
        cases=80,
        temperature=27,
        humidity=78,
        rainfall=120,
        population=569000,
        previous_cases=75,
        location="Dehradun",
        disease="Dengue",
        start_date="2026-10-04"
    )

    for forecast in forecasts:
        print(
            f"Day {forecast['day']} "
            f"| {forecast['date']} "
            f"| Predicted Cases: {forecast['predicted_cases']}"
        )

    print("=" * 50)