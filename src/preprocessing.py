import os
import numpy as np
import pandas as pd


def generate_dataset():
    np.random.seed(42)

    locations = [
        "Dehradun",
        "Haridwar",
        "Nainital",
        "Udham Singh Nagar",
        "Haldwani"
    ]

    diseases = [
        "Dengue",
        "Malaria",
        "Influenza"
    ]

    dates = pd.date_range(
        start="2022-01-02",
        end="2025-12-28",
        freq="W"
    )

    rows = []

    for location in locations:
        for disease in diseases:

            base_cases = np.random.randint(20, 80)

            for date in dates:

                month = date.month

                # Seasonal effect
                if month in [6, 7, 8, 9]:
                    seasonal_factor = 1.5
                elif month in [10, 11]:
                    seasonal_factor = 1.2
                else:
                    seasonal_factor = 0.8

                temperature = np.random.normal(
                    25 if month in [4, 5, 6, 7, 8] else 18,
                    3
                )

                humidity = np.random.normal(70, 10)

                rainfall = max(
                    0,
                    np.random.normal(
                        100 if month in [6, 7, 8, 9] else 20,
                        30
                    )
                )

                population = {
                    "Dehradun": 569000,
                    "Haridwar": 1890000,
                    "Nainital": 954000,
                    "Udham Singh Nagar": 1640000,
                    "Haldwani": 232000
                }[location]

                previous_cases = max(
                    1,
                    int(base_cases)
                )

                cases = (
                    base_cases
                    * seasonal_factor
                    + rainfall * 0.08
                    + humidity * 0.15
                    + np.random.normal(0, 8)
                )

                cases = max(0, int(cases))

                rows.append([
                    date,
                    location,
                    disease,
                    cases,
                    round(temperature, 2),
                    round(humidity, 2),
                    round(rainfall, 2),
                    population,
                    previous_cases
                ])

                base_cases = (
                    0.7 * base_cases
                    + 0.3 * cases
                )

    df = pd.DataFrame(
        rows,
        columns=[
            "date",
            "location",
            "disease",
            "cases",
            "temperature",
            "humidity",
            "rainfall",
            "population",
            "previous_cases"
        ]
    )

    output_directory = "data/raw"

    os.makedirs(output_directory, exist_ok=True)

    output_path = os.path.join(
        output_directory,
        "disease_data.csv"
    )

    df.to_csv(output_path, index=False)

    print(f"Dataset created: {output_path}")
    print(f"Rows: {len(df)}")
    print("\nFirst five rows:")
    print(df.head())


if __name__ == "__main__":
    generate_dataset()