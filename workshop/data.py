import pandas as pd
from pathlib import Path

from .config import default_config

DATA_DIR = Path() / "data"

def load_data(clean = True) -> pd.DataFrame:
    df = pd.read_csv(
        DATA_DIR / "Training_data" / "ALtrainingdata_workshop.csv",
        # Read all columns as text
        dtype=str,
    )

    if (clean):
        df = clean_data(df)

    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Replace every value of "9990" with na
    df = df.replace("9990", pd.NA)

    # Drop all rows with na in the column "isco88"
    df = df.dropna(subset=["isco88"])

    # Drop all rows where BOTH job_title and job_duties are NA
    df = df.dropna(subset=["job_title", "job_duties"], how="all")

    return df

def clean_freetext_col(col: pd.Series) -> pd.Series:
    # Remove all non-alphanumeric characters
    col = col.str.replace('[^0-9a-zA-Z.,-/!()+ ]', '', regex=True)

    return col

def preprocess_data(df: pd.DataFrame, config: dict = default_config) -> pd.DataFrame:
    # Combine job_title and job_duties into one column
    df["job_text"] = df["job_title"].fillna('') + " " + df["job_duties"].fillna('')
    # Drop the original columns
    df = df.drop(columns=["job_title", "job_duties"])

    df["job_text"] = clean_freetext_col(df["job_text"])

    # Classic text-preprocessing steps

    if config["capitalization"]:
        df["job_text"] = df["job_text"].str.lower()

    return df

