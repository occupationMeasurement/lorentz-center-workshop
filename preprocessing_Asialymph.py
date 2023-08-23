import pandas as pd
import re
import string
import zhon.hanzi as zh
import nltk
import spacy
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


def preprocess_text(text, to_lower=True, to_upper=False, remove_punctuation=True, remove_chinese=True, stem=False):
    if to_lower:
        text = text.lower()

    if to_upper:
        text = text.upper()

    if remove_punctuation:
        text = text.translate(str.maketrans("", "", string.punctuation))

    if remove_chinese:
        text = re.sub(rf"[{zh.characters}]", "", text)

    if stem:
        text = stem_text(text)

    return text


def stem_text(text):
    stemmer = PorterStemmer()
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(stemmed_tokens)


def load_and_preprocess_csv(file_path, min_combined_length=10, to_lower=True, to_upper=False, remove_punctuation=True,
                            remove_chinese=True, stem=False):
    try:
        data = pd.read_csv(file_path, dtype={'isco88': str})

        if 'job_title' in data.columns and 'job_duties' in data.columns and 'nci_trainingID' in data.columns and 'isco88' in data.columns:

            # Concatenate 'job_title' and 'job_duties'
            data['combined_text'] = data['job_title'].fillna('') + ' ' + data['job_duties'].fillna('')

            # Remove rows where combined_text is empty
            data = data[data['combined_text'].str.strip() != '']

            # Filter out rows with combined_text length less than min_combined_length
            data = data[data['combined_text'].str.len() >= min_combined_length]

            # Apply preprocessing to the combined text
            data['combined_text'] = data['combined_text'].apply(
                lambda x: preprocess_text(x, to_lower, to_upper, remove_punctuation, remove_chinese, stem))

            return data
        else:
            raise ValueError("The CSV must contain columns 'job_title', 'job_duties', 'nci_trainingID', and 'isco88'.")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# Example usage
csv_file_path = 'ALtrainingdata_workshop.csv'
min_length = 4  # Adjust the minimum combined text length as needed
preprocessed_data = load_and_preprocess_csv(csv_file_path, min_combined_length=min_length, to_lower=False, to_upper=False,
                                            remove_punctuation=True, remove_chinese=True, stem=False)