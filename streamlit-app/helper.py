import re
import os
import pickle
import numpy as np
import distance

from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz


# ==================================================
# File Paths (Safe Loading)
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STOPWORDS_PATH = os.path.join(BASE_DIR, "stopwords.pkl")
CV_PATH = os.path.join(BASE_DIR, "cv.pkl")


# ==================================================
# Load Files Once (Fast)
# ==================================================

STOP_WORDS = pickle.load(open(STOPWORDS_PATH, "rb"))
cv = pickle.load(open(CV_PATH, "rb"))


# ==================================================
# Contractions Dictionary
# ==================================================

CONTRACTIONS = {
    "ain't": "am not", "aren't": "are not", "can't": "can not",
    "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
    "don't": "do not", "hadn't": "had not", "hasn't": "has not",
    "haven't": "have not", "he's": "he is", "i'm": "i am",
    "isn't": "is not", "it's": "it is", "let's": "let us",
    "shouldn't": "should not", "that's": "that is", "there's": "there is",
    "they're": "they are", "wasn't": "was not", "we're": "we are",
    "weren't": "were not", "what's": "what is", "who's": "who is",
    "won't": "will not", "wouldn't": "would not", "you're": "you are"
}


# ==================================================
# Basic Features
# ==================================================

def test_common_words(q1, q2):
    return len(set(q1.split()) & set(q2.split()))


def test_total_words(q1, q2):
    return len(q1.split()) + len(q2.split())


# ==================================================
# Token Features
# ==================================================

def test_fetch_token_features(q1, q2):

    SAFE_DIV = 0.0001
    token_features = [0.0] * 8

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if not q1_tokens or not q2_tokens:
        return token_features

    q1_words = set(w for w in q1_tokens if w not in STOP_WORDS)
    q2_words = set(w for w in q2_tokens if w not in STOP_WORDS)

    q1_stops = set(w for w in q1_tokens if w in STOP_WORDS)
    q2_stops = set(w for w in q2_tokens if w in STOP_WORDS)

    common_word = len(q1_words & q2_words)
    common_stop = len(q1_stops & q2_stops)
    common_token = len(set(q1_tokens) & set(q2_tokens))

    token_features[0] = common_word / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word / (max(len(q1_words), len(q2_words)) + SAFE_DIV)

    token_features[2] = common_stop / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)

    token_features[4] = common_token / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)

    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])

    return token_features


# ==================================================
# Length Features
# ==================================================

def test_fetch_length_features(q1, q2):

    length_features = [0.0] * 3

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if not q1_tokens or not q2_tokens:
        return length_features

    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
    length_features[1] = (len(q1_tokens) + len(q2_tokens)) / 2

    strs = list(distance.lcsubstrings(q1, q2))

    if strs:
        length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1)

    return length_features


# ==================================================
# Fuzzy Features
# ==================================================

def test_fetch_fuzzy_features(q1, q2):

    return [
        fuzz.QRatio(q1, q2),
        fuzz.partial_ratio(q1, q2),
        fuzz.token_sort_ratio(q1, q2),
        fuzz.token_set_ratio(q1, q2)
    ]


# ==================================================
# Text Preprocessing
# ==================================================

def preprocess(q):

    q = str(q).lower().strip()

    q = q.replace('%', ' percent ').replace('$', ' dollar ') \
         .replace('₹', ' rupee ').replace('€', ' euro ') \
         .replace('@', ' at ').replace('[math]', '')

    # Expand contractions
    words = []
    for w in q.split():
        if w in CONTRACTIONS:
            w = CONTRACTIONS[w]
        words.append(w)

    q = " ".join(words)

    # Remove HTML
    q = BeautifulSoup(q, "html.parser").get_text()

    # Remove special chars
    q = re.sub(r'\W', ' ', q)
    q = re.sub(r'\s+', ' ', q).strip()

    return q


# ==================================================
# Main Feature Generator
# ==================================================

def query_point_creator(q1, q2):

    input_query = []

    # Preprocess
    q1 = preprocess(q1)
    q2 = preprocess(q2)

    # Basic Features
    input_query.extend([
        len(q1),
        len(q2),
        len(q1.split()),
        len(q2.split())
    ])

    common = test_common_words(q1, q2)
    total = test_total_words(q1, q2)

    input_query.append(common)
    input_query.append(total)

    input_query.append(round(common / total, 2) if total else 0)

    # Advanced Features
    input_query.extend(test_fetch_token_features(q1, q2))
    input_query.extend(test_fetch_length_features(q1, q2))
    input_query.extend(test_fetch_fuzzy_features(q1, q2))

    # Bag of Words
    q1_bow = cv.transform([q1]).toarray()
    q2_bow = cv.transform([q2]).toarray()

    return np.hstack((
        np.array(input_query).reshape(1, 22),
        q1_bow,
        q2_bow
    ))