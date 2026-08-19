"""
Text cleaning / preprocessing shared by both training (train_model.py) and
inference (app.py) so the exact same transformation is applied every time.
"""
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data quietly (only runs once, then cached).
for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

_STOPWORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<.*?>")
_NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/HTML/punctuation/numbers, remove stopwords,
    and lemmatize. Returns a single cleaned string ready for TF-IDF."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = text.split()
    tokens = [
        _LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in _STOPWORDS and len(tok) > 2
    ]
    return " ".join(tokens)
