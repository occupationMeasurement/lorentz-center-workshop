import pandas as pd
import fasttext
import fasttext.util
from sklearn.feature_extraction.text import CountVectorizer

from .config import default_config

def get_embeddings(df: pd.DataFrame, config: dict = default_config) -> pd.DataFrame:
    """
    Turn text data into numerical representation (embeddings by default)

    This function will remove the original text column (combined_text) and add new
    columns corresponding to the dimensions of the embeddings.
    """

    if config["embeddings_engine"] == "fasttext":
        fasttext.util.download_model('en', if_exists='ignore')
        print("Loading Model.")
        ft_model = fasttext.load_model("cc.en.300.bin")
        print("Model Loaded.")

        df = df.apply(
            lambda row: ft_model.get_sentence_vector(row["combined_text"]),
            result_type='expand',
            axis=1,
        )
    elif config["embeddings_engine"] == "bag_of_words":
        # Create a bag of words (using default settings)
        count_vect = CountVectorizer()
        counts = count_vect.fit_transform(df["combined_text"])
        df = pd.DataFrame(counts.todense(), columns=count_vect.get_feature_names_out())


    # Drop the original text column (and the target)
    if "combined_text" in df.columns:
        df = df.drop(columns=["combined_text"])
    if "isco88" in df.columns:
        df = df.drop(columns=["isco88"])

    return df


