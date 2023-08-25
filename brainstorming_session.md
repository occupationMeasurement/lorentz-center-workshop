# Brainstorming session 

The ML-Pipeline inlcudes (1) Preprocessing, (2) generation of embeddings, (3) classification, and (4) Evaluation. In the brainstorming session we found the following to be relevant:

## Preprocessing

Whether and how to do preprocessing may depend on the later steps in the pipeline. 

- Capitalization (almost always recommended to upper-/lowercase text). This will depend on the type of algorithm that will be used. For example, BERT has both a cased and uncased version. For the cased version, the dictionary of words is extended to include capitalized words since they could represent different information compared to their uncased version. In such cases, one might consider not upper/lowercasing text. This will also depend on the language of the input descriptions, where capitalized words could have different meanings. 
- Remove stopwords (irrelevant with most algorithms)
- Remove punctuation (may depend on language)
- Spelling correction (Goal should be: standardize different spellings; unclear how to do it best, depends on data/language)
- String similarities (relates to spelling correction)
- Removal of redundant words. For example, if the job description is 'cleaner', and task is 'cleaning', one might consider either one 'redundant' and therefore considered 'noise' for the classification algorithm.
- Language detection (remove/translate Chinese characters, maybe standardize Umlaut and accent characters, ...) In the case of multiple languages in one dataset, one could consider standardizing to one language using machine translation. Several problems with this however: 1) How to consistently recognize different languages, and 2) The validity of the machine translation, where literal translations of a description in one language might not represent the same job in another language.

For tokenization one may consider the following (unclear what is best):

- lemmatization
- stemming
- hyphenation
- n-grams

The usage of either lemmatization or stemming will also depend on the following ML/DL model that will be used. For example, pre-trained LLMs (e.g., BERT) are usually trained on large volumes of data with complete words. If inputs are stemmed/lemmatized, this type of model might not recognize such words to the same extent that it would recognize (and 'correctly' embed) full words on which it was trained. Therefore, one might consider not using these types of pre-processing strategies in such a scenario. However, in the case of usage with e.g., bag of words, one might benefit from increased standardization. This could possibly remove irrelevant 'noisy' words, limiting the vocabulary and therefore possibly improving generalization and subsequent model performance.

### Special considerations for occupation data

- standardize gendered job titles? (e.g. politician, in German: Politiker (m), Politikerin (f))  --> also ties into stemming/lemmatization
- remove text if the same job title is mentioned multiple times? --> could potentially help with the inbalance in the number of available occupational codes in the databases.
- Concatenate text fields "job title" and "job tasks", or do we lose relevant information this way, for example if coders have a clear priority to use the job title and mostly ignore the job tasks. 
- standardize gendered job titles? (e.g. politician, in German: Politiker (m), Politikerin (f))
- remove text if the same job title is mentioned multiple times? (tested: made results worse)
- Concatenate text fields "job title" and "job tasks", or do we loose relevant information this way, for example if coders have a clear priority to use the job title and mostly ignore the job tasks.

Dealing with extra codes:

- General codes ending in 0
- Extra codes (99xx) for housewife, student, retired, ...

One may want to remove these codes because they are not in the official classification, or keep them because they are relevant for the respective application.

Usage of external data will probably improve predictions?

- Coding index
- O*Net

This is especially helpful in increasing the representation of minority classes, where only one or two entries are present for certain occupational codes. Additionally, since the examples are database-independent (i.e., not described in a certain way resulting from a certain type of question or input format), this could potentially improve generalizability.

## Embeddings

Possibilities include:

- Bag-of-Words
- fasttext
- Embeddings from BERT-like models
- OpenGPT
- Llama
- Embedding models fine tuned to occupation data e.g. https://github.com/junhua/ipod (or probably a more modern version of these)

## Classification

We distinguish between classical machine learning and deep learning.

How to best split the dataset for parameter tuning? Unlcear. In all cases, one should consider using a validation set during training to ensure the model does not overfit on the training data (e.g., using early stopping when validation loss does not decrease). This could improve generalizability.

### Classical Machine Learning

- kNN (fast and simple, but probably not the best. See Gweon et al. 2017)
- SVM
- Logistic Regression
- Boosting (XGBoost) (extensive tuning needed, but results are often better)

### Deep Learning

- CNN (or other architectures?)

Considerations for tuning:

- Finetuning
- Freezing Layers?

Unfreezing layers could potentially improve within distribution/database performance, but could decrease generalization capabilities (i.e., out-of-distribution performance). One could view the number of frozen layers as a hyperparameter, and determine its effect on the within and outside of distribution performance.

## Evaluation

More than one category can often be considered correct: Coders may select multiple appropriate categories if asked to. This is called "multi-label classification". Should we think about occupation coding in these terms? While we endorse this approach, we the following is about single-label evaluation.

Different visions what we would want to know during evalution:

- How well does the system perform for every possible input (or at least for the entirety of possible codes?)
- How well does the system perform in the (future) study population? (which is especially in epidemiology often unknown)

### Metrics

Difficult to choose - and all metrics have their issues.

- Accuracy: raw / averaged over categories / for each major group
- F1: Micro / Macro average

How can we make use of predicted "probabilities"?
- AUC (but what does this look like for multi-class classification?)
- Reliability diagram (plot accuracy vs predicted probability)
- Accuracy at different thresholds (maybe as a plot)
- Cross-Entropy (?)

Are additional metrics helpful if they are directly related to the coding system or the application?

In general we recommend using diagrams. Maybe this allows to show all accuracies (in different colors) for all category at each level of the hierarchy.
