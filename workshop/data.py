import pandas as pd
from pathlib import Path

from .config import default_config
from .preprocessing_Asialymph import load_and_preprocess_csv

DATA_DIR = Path() / "data"

def load_and_preprocess_data(config: dict = default_config) -> pd.DataFrame:
    df = load_and_preprocess_csv(
        DATA_DIR / "Training_data" / "ALtrainingdata_workshop.csv"
    )

    return df
