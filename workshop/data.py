import pandas as pd
from pathlib import Path

from .config import default_config
from .preprocessing_Asialymph import load_and_preprocess_csv, preprocess_text

ROOT_DIR = Path()
DATA_DIR = ROOT_DIR / "data"


def load_and_preprocess_data(config: dict = default_config) -> pd.DataFrame:
    return load_and_preprocess_csv(
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


def load_and_preprocess_validation_data(config: dict = default_config) -> pd.DataFrame:
    return load_and_preprocess_csv(
        DATA_DIR / "Validation_data" / "ALvalidationdata_workshop_nocode.csv",
        min_combined_length=config["min_combined_length"],
        to_lower=config["to_lower"],
        to_upper=config["to_upper"],
        remove_punctuation=config["remove_punctuation"],
        remove_chinese=config["remove_chinese"],
        stem=config["stem"],
        # Don't discard entries
        only_4digit=False,
        only_exist=False,
    )


def load_isco88_structure(config: dict) -> pd.DataFrame:
    isco = pd.read_excel(
        ROOT_DIR / "isco" / "ISCO-88 EN Structure and defnitions.xlsx", dtype=str
    )
    isco = isco.rename(
        columns={
            "ISCO 88 Code": "isco88",
            "Included occupations": "occupations",
        }
    )[["isco88", "occupations"]]

    # Remove everything after the colon
    isco["occupations"] = isco["occupations"].str.split(":").str[1]
    # Turn all text with newlines into multiple rows
    isco = isco.assign(occupations=isco.occupations.str.split("\n")).explode(
        "occupations"
    )
    # Only keep rows with 4 digits
    isco = isco[isco["isco88"].str.match(r"^\d{4}")]
    # Remove leading dashes
    isco["occupations"] = isco["occupations"].str.replace("-", " ")
    # Strip whitespace
    isco["occupations"] = isco["occupations"].str.strip()
    # Remove empty rows
    isco = isco[isco["occupations"].fillna("") != ""]

    isco["occupations"] = isco["occupations"].apply(
        lambda x: preprocess_text(
            x, config["to_lower"], config["to_upper"], config["remove_punctuation"], config["remove_chinese"], config["stem"]
        )
    )

    return isco
