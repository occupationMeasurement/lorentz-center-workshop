import pandas as pd
from pathlib import Path

from .config import default_config
from .preprocessing_Asialymph import load_and_preprocess_csv

DATA_DIR = Path() / "data"

def load_and_preprocess_data(config: dict = default_config) -> pd.DataFrame:
    df = load_and_preprocess_csv(
        DATA_DIR / "Training_data" / "ALtrainingdata_workshop.csv",

        min_combined_length=config["min_combined_length"],
        to_lower=config["to_lower"],
        to_upper=config["to_upper"],
        remove_punctuation=config["remove_punctuation"],
        remove_chinese=config["remove_chinese"],
        stem=config["stem"],
        only_4digit=config["only_4digit"],
        only_exist=config["only_exist"],
    )

    return df
