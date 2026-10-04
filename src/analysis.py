import pandas as pd
import matplotlib.pyplot as plt
import os


DATA_PATH = "data/raw/disease_data.csv"
OUTPUT_DIR = "data/processed/plots"


def load_data():
    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"])

    return df


def create_output_directory():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def disease_distribution(df):
    counts = df.groupby("disease")["cases"].sum()

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")

    plt.title("Total Disease Cases")
    plt.xlabel("Disease")
    plt.ylabel("Total Cases")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/disease_distribution.png"
    )

    plt.close()


def cases_over_time(df):
    weekly_cases = (
        df.groupby("date")["cases"]
        .sum()
    )

    plt.figure(figsize=(12, 5))

    weekly_cases.plot()

    plt.title("Disease Cases Over Time")
    plt.xlabel("Date")
    plt.ylabel("Cases")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/cases_over_time.png"
    )

    plt.close()


def temperature_vs_cases(df):
    plt.figure(figsize=(8, 5))

    plt.scatter(
        df["temperature"],
        df["cases"],
        alpha=0.3
    )

    plt.title("Temperature vs Disease Cases")
    plt.xlabel("Temperature")
    plt.ylabel("Cases")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/temperature_vs_cases.png"
    )

    plt.close()


def humidity_vs_cases(df):
    plt.figure(figsize=(8, 5))

    plt.scatter(
        df["humidity"],
        df["cases"],
        alpha=0.3
    )

    plt.title("Humidity vs Disease Cases")
    plt.xlabel("Humidity")
    plt.ylabel("Cases")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/humidity_vs_cases.png"
    )

    plt.close()


def rainfall_vs_cases(df):
    plt.figure(figsize=(8, 5))

    plt.scatter(
        df["rainfall"],
        df["cases"],
        alpha=0.3
    )

    plt.title("Rainfall vs Disease Cases")
    plt.xlabel("Rainfall")
    plt.ylabel("Cases")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/rainfall_vs_cases.png"
    )

    plt.close()


def location_analysis(df):
    location_cases = (
        df.groupby("location")["cases"]
        .sum()
        .sort_values()
    )

    plt.figure(figsize=(10, 6))

    location_cases.plot(kind="barh")

    plt.title("Total Cases by Location")
    plt.xlabel("Total Cases")
    plt.ylabel("Location")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/cases_by_location.png"
    )

    plt.close()


def main():
    print("Loading dataset...")

    df = load_data()

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDisease counts:")
    print(df["disease"].value_counts())

    print("\nLocation counts:")
    print(df["location"].value_counts())

    create_output_directory()

    disease_distribution(df)
    cases_over_time(df)
    temperature_vs_cases(df)
    humidity_vs_cases(df)
    rainfall_vs_cases(df)
    location_analysis(df)

    print("\nAnalysis completed.")
    print(f"Plots saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()