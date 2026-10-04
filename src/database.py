import sqlite3
from pathlib import Path
from datetime import datetime


# =========================================================
# DATABASE PATH
# =========================================================

DATABASE_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "disease_predictions.db"
)


# =========================================================
# DATABASE CONNECTION
# =========================================================

def get_connection():

    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


# =========================================================
# INITIALIZE DATABASE
# =========================================================

def init_database():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            prediction_date TEXT NOT NULL,

            forecast_date TEXT,

            disease TEXT NOT NULL,

            location TEXT NOT NULL,

            current_cases REAL NOT NULL,

            previous_cases REAL NOT NULL,

            temperature REAL NOT NULL,

            humidity REAL NOT NULL,

            rainfall REAL NOT NULL,

            population REAL NOT NULL,

            predicted_cases REAL NOT NULL,

            growth_rate REAL NOT NULL,

            historical_mean REAL NOT NULL,

            risk_score REAL NOT NULL,

            risk_level TEXT NOT NULL,

            case_level TEXT NOT NULL,

            growth_level TEXT NOT NULL
        )
        """
    )

    connection.commit()


    # -----------------------------------------------------
    # Upgrade existing database
    # -----------------------------------------------------

    cursor.execute(
        "PRAGMA table_info(predictions)"
    )

    columns = [
        row["name"]
        for row in cursor.fetchall()
    ]


    if "forecast_date" not in columns:

        cursor.execute(
            """
            ALTER TABLE predictions
            ADD COLUMN forecast_date TEXT
            """
        )

        connection.commit()


    connection.close()


# =========================================================
# SAVE PREDICTION
# =========================================================

def save_prediction(
    disease,
    location,
    current_cases,
    previous_cases,
    temperature,
    humidity,
    rainfall,
    population,
    predicted_cases,
    growth_rate,
    historical_mean,
    risk_score,
    risk_level,
    case_level,
    growth_level,
    forecast_date=None
):

    connection = get_connection()

    cursor = connection.cursor()


    prediction_date = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


    cursor.execute(
        """
        INSERT INTO predictions (

            prediction_date,

            forecast_date,

            disease,

            location,

            current_cases,

            previous_cases,

            temperature,

            humidity,

            rainfall,

            population,

            predicted_cases,

            growth_rate,

            historical_mean,

            risk_score,

            risk_level,

            case_level,

            growth_level

        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,

        (
            prediction_date,

            forecast_date,

            disease,

            location,

            current_cases,

            previous_cases,

            temperature,

            humidity,

            rainfall,

            population,

            predicted_cases,

            growth_rate,

            historical_mean,

            risk_score,

            risk_level,

            case_level,

            growth_level
        )
    )


    connection.commit()

    connection.close()


# =========================================================
# GET PREDICTION HISTORY
# =========================================================

def get_prediction_history():

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT *

        FROM predictions

        ORDER BY id DESC
        """
    )


    records = cursor.fetchall()

    connection.close()


    return records


# =========================================================
# TEST DATABASE
# =========================================================

if __name__ == "__main__":

    init_database()

    print(
        "Database initialized successfully."
    )

    print(
        f"Database location:\n{DATABASE_PATH}"
    )