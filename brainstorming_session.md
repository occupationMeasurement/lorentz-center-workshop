# Brainstorming session 

The ML-Pipeline inlcudes (1) Preprocessing, (2) generation of embeddings, (3) classification, and (4) Evaluation. In the brainstorming session we found the following to be relevant:

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
- remove text if the same job title is mentioned multiple times? (tested: made results worse)
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