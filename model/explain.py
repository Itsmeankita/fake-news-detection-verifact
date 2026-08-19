"""
Lightweight explainability for the deployed model.

For a linear model (Logistic Regression / Passive Aggressive), each TF-IDF
feature has a learned weight. Multiplying a document's TF-IDF value for a
word by that word's model weight tells us how much that specific word
pushed the prediction toward FAKE (positive) or REAL (negative). This gives
"why did the model say this" transparency without needing a heavy library
like SHAP/LIME — appropriate for a linear bag-of-words model.
"""
import numpy as np


def top_contributing_words(model, vectorizer, vec, top_n=8):
    """Returns two lists: words pushing toward FAKE, words pushing toward REAL.
    Falls back to an empty explanation for models without linear coefficients
    (e.g. Naive Bayes) since the same trick doesn't apply directly."""
    if not hasattr(model, "coef_"):
        return [], []

    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]  # positive => pushes toward the "greater" class index

    # sklearn sorts classes alphabetically: FAKE=0, REAL=1 -> positive coef pushes REAL
    row = vec.tocoo()
    contributions = {}
    for col, value in zip(row.col, row.data):
        contributions[col] = value * coefs[col]

    if not contributions:
        return [], []

    sorted_items = sorted(contributions.items(), key=lambda x: x[1])

    fake_pushing = [feature_names[i] for i, v in sorted_items if v < 0][:top_n]
    real_pushing = [feature_names[i] for i, v in reversed(sorted_items) if v > 0][:top_n]

    return fake_pushing, real_pushing


def global_top_words(model, vectorizer, top_n=15):
    """Model-wide (not per-request) explainability: the words with the
    strongest learned weight toward each class overall. Used on the
    Analytics page to show what the model 'pays attention to' in general."""
    if not hasattr(model, "coef_"):
        return [], []

    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]
    order = np.argsort(coefs)  # ascending: most negative (FAKE-pushing) first

    fake_words = [(feature_names[i], round(float(coefs[i]), 3)) for i in order[:top_n]]
    real_words = [(feature_names[i], round(float(coefs[i]), 3)) for i in order[::-1][:top_n]]
    return fake_words, real_words
