import pandas as pd
import re
import string
import zhon.hanzi as zh
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split


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


def load_and_preprocess_csv(file_path, min_combined_length=10, to_lower=True, to_upper=False,
                            remove_punctuation=True, remove_chinese=True, stem=False, only_4digit=False, only_exist=False):
    try:
        data = pd.read_csv(file_path, dtype={'isco88': str})

        if 'job_title' in data.columns and 'job_duties' in data.columns and ('nci_trainingID' in data.columns or 'ID' in data.columns) and 'isco88' in data.columns:
            data['combined_text'] = data['job_title'].fillna('') + ' ' + data['job_duties'].fillna('')
            data = data[data['combined_text'].str.strip() != '']
            data = data[data['combined_text'].str.len() >= min_combined_length]
            data['combined_text'] = data['combined_text'].apply(lambda x: preprocess_text(x, to_lower, to_upper,
                                                                                          remove_punctuation,
                                                                                          remove_chinese, stem))
            if only_exist:
                coding_index = pd.read_excel("ISCO88_all_groups.xlsx", dtype={'ISCO88_4D': str})
                valid_isco88_codes = set(coding_index['ISCO88_4D'])
                data = data[data['isco88'].isin(valid_isco88_codes)]

            if only_4digit:
                # Load coding index
                coding_index = pd.read_excel("ISCO88_all_groups.xlsx", dtype={'ISCO88': str})
                coding_index = coding_index[coding_index['ISCO88'].str.match('^\d{4}$', na=False)]
                valid_isco88_codes = set(coding_index['ISCO88'])
                data = data[data['isco88'].isin(valid_isco88_codes)]

            return data
        else:
            raise ValueError("The CSV must contain columns 'job_title', 'job_duties', 'nci_trainingID' (or ID), and 'isco88'.")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def train_split(dataset, split_size = 0.2, stratify = True):
    try:
        if 'isco88' in dataset.columns:
            if stratify:
                # Filter classes with 1 occurrence
                dataset = dataset.groupby(['isco88']).filter(lambda group: len(group)>1)
                train, test = train_test_split(dataset, 
                                           test_size=split_size, random_state=3, 
                                           stratify=dataset.loc[:,'isco88'])
                return train, test
            else:
                train, test = train_test_split(dataset, 
                                           test_size=split_size, random_state=3, 
                                           stratify=dataset.loc[:,'isco88'])
                return train, test
        else:
            raise ValueError("The CSV must contain column 'isco88'.")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# Example usage
#csv_file_path = 'ALtrainingdata_workshop.csv'
# min_length = 3
# preprocessed_data_all = load_and_preprocess_csv(csv_file_path, min_combined_length=min_length,
#                                             to_lower=True, to_upper=False,
#                                             remove_punctuation=True, remove_chinese=True, stem=False, only_4digit=True,
#                                             only_exist=True)

# test, train = train_split(preprocessed_data_all)
