# Brainstorming session

The ML-Pipeline inlcudes (1) Preprocessing, (2) generation of embeddings, and (3) classification. In the brainstorming session we found the following to be relevant:

## Preprocessing

Whether and how to do preprocessing may depend on the later steps in the pipeline.

- Capitalization (always recommended to upper-/lowercase text)
- Remove stopwords (irrelevant with most algorithms)
- Remove punctuation (may depend on language)
- Spelling correction (Goal should be: standardize different spellings; unclear how to do it best, depends on data/language)
- String similarities (relates to spelling correction)
- Language detection (remove/translate Chinese characters, maybe standardize Umlaut and accent characters, ...)

For tokenization one may consider the following (unclear what is best):

- lemmatization, 
- stemming,
- hyphenization
- n-grams

### Special considerations for occupation data

- standardize gendered job titles? (e.g. politician, in German: Politiker (m), Politikerin (f))
- remove text if the same job title is mentioned multiple times?
- Concatenate text fields "job title" and "job tasks", or do we loose relevant information this way, for example if coders have a clear priority to use the job title and mostly ignore the job tasks.

Dealing with extra codes:

- General codes ending in 0
- Extra codes (99xx) for housewife, student, retired, ...

One may want to remove these codes because they are not in the official classification, or keep them because they are relevant for the respective application.

Usage of external data will probably improve predictions?

- Coding index
- O*Net

## Embeddings

Possibilities include:

- Bag-of-Words
- fasttext
- Embeddings from BERD-like models
- OpenGPT
- Llama

## Classification

We distinguish between classical machine learning and deep learning.

How to best split the dataset for parameter tuning? Unlcear.

### Classical Machine Learning

- kNN (fast and simple, but probably not the best. See Gweon et al. 2017)
- SVM
- Logistic Regression
- Boosting (XgBoost) (extensive tuning needed, but results are often better)

### Deep Learning

- CNN (or other architectures?)

Considerations for tuning:

- Finetuning
- Freezing Layers?