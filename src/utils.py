# src/utils.py
import re
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
from typing import List

STOPWORDS = set(stopwords.words('english'))

def clean_tweet(text: str, remove_hashtags=True, remove_mentions=True, remove_urls=True, lower=True) -> str:
    if not isinstance(text, str):
        return ""
    s = text
    if remove_urls:
        s = re.sub(r'http\S+|www\.\S+', '', s)
    if remove_mentions:
        s = re.sub(r'@\w+', '', s)
    if remove_hashtags:
        s = re.sub(r'#\w+', '', s)
    # remove punctuation
    s = s.translate(str.maketrans('', '', string.punctuation))
    if lower:
        s = s.lower()
    # collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def tokenize_and_remove_stopwords(text: str) -> List[str]:
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.lower() not in STOPWORDS and t.isalpha()]
    return tokens

