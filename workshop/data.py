import pandas as pd
from pathlib import Path

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

    df["job_title"] = clean_freetext_col(df["job_title"])
    df["job_duties"] = clean_freetext_col(df["job_duties"])

    return df

def clean_freetext_col(col: pd.Series) -> pd.Series:

    print("Test")

    # Remove all non-alphanumeric characters
    col = col.str.replace('[^0-9a-zA-Z.,-/!()+ ]', '', regex=True)

    return col

