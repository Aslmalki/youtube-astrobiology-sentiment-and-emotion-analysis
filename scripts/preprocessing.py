"""
Preprocessing utilities for sentiment and emotion analysis.
Extracted from the multi-method analysis pipeline.
"""

import re
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def preprocess_for_textblob(text):
    """
    Heavy preprocessing for TextBlob: lowercase, remove punctuation/emojis, lemmatize.
    """
    if not text or str(text).strip() == '':
        return ''
    text = str(text).strip()
    text = contractions.fix(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    emoji_pattern = re.compile("["
        r"\U0001F600-\U0001F64F"
        r"\U0001F300-\U0001F5FF"
        r"\U0001F680-\U0001F6FF"
        r"\U0001F1E0-\U0001F1FF"
        r"\u2600-\u26FF"
        r"\u2700-\u27BF"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t.lower() not in stop_words and len(t) > 1]
    return ' '.join(tokens)


def preprocess_for_vader(text):
    """
    Minimal preprocessing for VADER: preserve punctuation, caps, emojis.
    Only remove URLs and mentions.
    """
    if not text or str(text).strip() == '':
        return ''
    text = str(text).strip()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    return text


def preprocess_for_transformer(text):
    """
    Light preprocessing for Transformer models: remove URLs and mentions only.
    """
    if not text or str(text).strip() == '':
        return ''
    text = str(text).strip()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    return text
