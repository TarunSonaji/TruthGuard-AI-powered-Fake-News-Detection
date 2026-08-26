"""
utils/text_preprocessing.py
----------------------------
NLP preprocessing pipeline used by BOTH train_model.py and predict.py so that
training and inference always see text cleaned the exact same way.

Pipeline steps:
    1. Lowercase conversion
    2. Remove URLs
    3. Remove punctuation & special characters
    4. Remove numbers
    5. Tokenization
    6. Stop word removal
    7. Lemmatization

The module tries to use NLTK's data files (punkt, stopwords, wordnet).
If those data files are not available (e.g. no internet access) it falls
back to a small built-in English stop word list and skips lemmatization,
so the app still works out of the box in restricted environments.
"""

import re
import string

# --------------------------------------------------------------------------
# Try to set up NLTK. Fall back gracefully if data/packages are missing.
# --------------------------------------------------------------------------
_NLTK_READY = False
try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    # Make sure the required corpora are present. This will attempt a
    # download only once; if there's no internet connection it silently
    # fails and we drop into the fallback branch below.
    for resource in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.data.find(
                f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}"
            )
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass

    STOP_WORDS = set(nltk_stopwords.words("english"))
    _LEMMATIZER = WordNetLemmatizer()
    _NLTK_READY = True
except Exception:
    _NLTK_READY = False

# --------------------------------------------------------------------------
# Fallback stop word list (used only if NLTK data isn't available)
# --------------------------------------------------------------------------
_FALLBACK_STOP_WORDS = set("""
a about above after again against all am an and any are aren't as at be
because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for
from further had hadn't has hasn't have haven't having he he'd he'll he's
her here here's hers herself him himself his how how's i i'd i'll i'm i've
if in into is isn't it it's its itself let's me more most mustn't my myself
no nor not of off on once only or other ought our ours ourselves out over
own same shan't she she'd she'll she's should shouldn't so some such than
that that's the their theirs them themselves then there there's these they
they'd they'll they're they've this those through to too under until up
very was wasn't we we'd we'll we're we've were weren't what what's when
when's where where's which while who who's whom why why's with won't would
wouldn't you you'd you'll you're you've your yours yourself yourselves said
also one two says news report reported according
""".split())

if not _NLTK_READY:
    STOP_WORDS = _FALLBACK_STOP_WORDS


def _simple_tokenize(text: str):
    """Basic whitespace tokenizer used when NLTK's tokenizer isn't ready."""
    return text.split()


def clean_text(text: str) -> str:
    """
    Run the full preprocessing pipeline on a raw string and return a single
    cleaned string ready to be fed into the TF-IDF vectorizer.
    """
    if not isinstance(text, str):
        text = str(text)

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 3. Remove HTML tags (common in scraped news datasets)
    text = re.sub(r"<.*?>", " ", text)

    # 4. Remove numbers
    text = re.sub(r"\d+", " ", text)

    # 5. Remove punctuation / special characters
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-z\s]", " ", text)

    # 6. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 7. Tokenization
    if _NLTK_READY:
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = _simple_tokenize(text)
    else:
        tokens = _simple_tokenize(text)

    # 8. Remove stop words + very short tokens
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    # 9. Lemmatization
    if _NLTK_READY:
        try:
            tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]
        except Exception:
            pass

    return " ".join(tokens)


def get_suspicious_words(text: str, vectorizer, model, top_n: int = 8):
    """
    Heuristic explainability helper: returns the words in the given text
    that carry the strongest weight toward the "Fake" class, according to
    the linear model's learned coefficients.

    Works for linear models exposing `coef_` (Logistic Regression, Linear
    SVM, Passive Aggressive). For models without coefficients (e.g. Naive
    Bayes with certain configurations) it falls back to TF-IDF weight.
    """
    cleaned = clean_text(text)
    words = list(dict.fromkeys(cleaned.split()))  # unique, order-preserving
    if not words:
        return []

    vocab = vectorizer.vocabulary_
    scores = []

    coef = getattr(model, "coef_", None)
    if coef is not None:
        coef = coef[0]
        for w in words:
            idx = vocab.get(w)
            if idx is not None and idx < len(coef):
                scores.append((w, coef[idx]))
        # Highest positive coefficient == pushes toward "Fake" (class 0)
        # since Fake=0 / Real=1, a NEGATIVE coefficient pushes toward Fake.
        scores.sort(key=lambda x: x[1])
        suspicious = [w for w, s in scores[:top_n] if s < 0]
        return suspicious

    # Fallback: just return the highest TF-IDF weighted words
    tfidf_vec = vectorizer.transform([cleaned]).toarray()[0]
    for w in words:
        idx = vocab.get(w)
        if idx is not None:
            scores.append((w, tfidf_vec[idx]))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [w for w, s in scores[:top_n]]
