from src.predict import predict_7_days
from src.risk_engine import load_historical_data, calculate_risk


def run_prediction(
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
    Generate a 7-day disease forecast and calculate
    risk information for every forecast day.
    """

    # Generate 7-day predictions
    forecasts = predict_7_days(
        cases=cases,
        temperature=temperature,
        humidity=humidity,
        rainfall=rainfall,
        population=population,
        previous_cases=previous_cases,
        location=location,
        disease=disease,
        start_date=forecast_date
    )

    if not forecasts:
        raise ValueError(
            "Unable to generate forecast."
        )

    # Load historical data
    historical_data = load_historical_data()

    # Filter data for selected location and disease
    filtered_data = historical_data[
        (historical_data["location"] == location) &
        (historical_data["disease"] == disease)
    ]

    if filtered_data.empty:
        raise ValueError(
            f"No historical data found for "
            f"{disease} in {location}."
        )

    # Historical case values
    historical_cases = filtered_data["cases"]

    # Calculate historical statistics
    historical_mean = float(
        historical_cases.mean()
    )

    historical_std = float(
        historical_cases.std()
    )

    forecast_results = []

    # Calculate risk for every forecast day
    for forecast in forecasts:

        risk_result = calculate_risk(
            current_cases=cases,
            predicted_cases=forecast["predicted_cases"],
            historical_cases=historical_cases
        )

        forecast_results.append({

            "day":
                forecast["day"],

            "date":
                forecast["date"],

            "predicted_cases":
                forecast["predicted_cases"],

            "growth_rate":
                round(
                    risk_result["growth_rate"],
                    2
                ),

            "risk_score":
                round(
                    risk_result["risk_score"],
                    2
                ),

            "risk":
                risk_result["risk"],

            "case_level":
                risk_result["case_level"],

            "growth_level":
                risk_result["growth_level"],

            "historical_mean":
                round(
                    historical_mean,
                    2
                ),

            "historical_std":
                round(
                    historical_std,
                    2
                )

        })

    # Extract predicted case values
    predicted_values = [
        item["predicted_cases"]
        for item in forecast_results
    ]

    # Average prediction
    average_predicted_cases = (
        sum(predicted_values)
        / len(predicted_values)
    )

    # Maximum prediction
    maximum_predicted_cases = max(
        predicted_values
    )

    # Highest-risk day
    highest_risk_day = max(
        forecast_results,
        key=lambda item:
        item["risk_score"]
    )

    # Return complete result
    return {

        "forecast_date":
            str(forecast_date),

        "historical_mean":
            round(
                historical_mean,
                2
            ),

        "historical_std":
            round(
                historical_std,
                2
            ),

        "forecasts":
            forecast_results,

        "average_predicted_cases":
            round(
                average_predicted_cases,
                2
            ),

        "maximum_predicted_cases":
            round(
                maximum_predicted_cases,
                2
            ),

        "highest_risk_day":
            highest_risk_day["date"],

        "highest_risk":
            highest_risk_day["risk"],

        "highest_risk_score":
            highest_risk_day["risk_score"]

    }