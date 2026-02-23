#!/usr/bin/env python3
"""
Execute Notebook 3: Preprocessing and Feature Engineering
Extracted code from notebook cells
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'notebooks'))

# Change to notebooks directory for relative paths
os.chdir(os.path.join(os.path.dirname(__file__), 'notebooks'))

# Cell 1: Imports
import pandas as pd
import numpy as np
import re
import contractions
import warnings
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)

np.random.seed(42)
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')

print("Libraries imported successfully")
print("NLTK data downloaded")

# Load cleaned English dataset
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

df = pd.read_csv('../data/processed/01_comments_english.csv')
print(f"Loaded dataset: {len(df):,} comments, {len(df.columns)} columns")

# Ensure comment_text_original exists
if 'comment_text_original' not in df.columns:
    raise ValueError("comment_text_original column not found in dataset")

# Create text_raw column (no preprocessing)
df['text_raw'] = df['comment_text_original'].astype(str)
print(f"✓ Created text_raw column")

# Preprocessing functions
def preprocess_for_textblob(text):
    """
    Heavy preprocessing for TextBlob.
    Removes punctuation, emojis, stopwords, and applies lemmatization.
    """
    if pd.isna(text) or str(text).strip() == '':
        return ""
    
    text = str(text)
    
    # 1. Expand contractions
    try:
        text = contractions.fix(text)
    except:
        pass
    
    # 2. Lowercase
    text = text.lower()
    
    # 3. Remove URLs, mentions, hashtags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    
    # 4. Remove special characters and emojis
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # 5. Tokenize
    try:
        tokens = word_tokenize(text)
    except:
        tokens = text.split()
    
    # 6. Remove stopwords
    try:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
    except:
        pass
    
    # 7. Lemmatization
    try:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    except:
        pass
    
    # 8. Join and clean
    text = ' '.join(tokens).strip()
    
    return text

def preprocess_for_vader(text):
    """
    Minimal preprocessing for VADER.
    
    CRITICAL: Keep punctuation, caps, emojis - VADER needs these!
    """
    if pd.isna(text) or str(text).strip() == '':
        return ""
    
    text = str(text)
    
    # ONLY remove URLs and mentions
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    
    # Clean whitespace
    text = ' '.join(text.split())
    
    # DO NOT remove: punctuation, caps, emojis, hashtags
    
    return text.strip()

def preprocess_for_transformer(text):
    """
    Light preprocessing for Transformers.
    Keep structure, remove only noise.
    """
    if pd.isna(text) or str(text).strip() == '':
        return ""
    
    text = str(text)
    
    # Remove URLs and mentions
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    
    # Clean whitespace
    text = ' '.join(text.split())
    
    # Keep: punctuation, caps, emojis, structure
    
    return text.strip()

print("Preprocessing functions defined")

# Apply all preprocessing tracks
print("=" * 60)
print("CREATING FOUR PREPROCESSING TRACKS")
print("=" * 60)

# Enable progress bar
tqdm.pandas(desc="Processing")

# Track A: Heavy preprocessing for TextBlob
print("\n1. Creating text_textblob (heavy preprocessing)...")
df['text_textblob'] = df['text_raw'].progress_apply(preprocess_for_textblob)

# Track B: Minimal preprocessing for VADER
print("\n2. Creating text_vader (minimal preprocessing - KEEPS punctuation, caps, emojis)...")
df['text_vader'] = df['text_raw'].progress_apply(preprocess_for_vader)

# Track C: Light preprocessing for Transformers
print("\n3. Creating text_transformer (light preprocessing)...")
df['text_transformer'] = df['text_raw'].progress_apply(preprocess_for_transformer)

# Track D: text_raw already exists (no preprocessing)
print("\n4. text_raw preserved for engagement features")

print("\n✓ All four preprocessing tracks created")
print(f"  text_raw: {len(df[df['text_raw'].str.strip() != ''])}, non-empty texts")
print(f"  text_textblob: {len(df[df['text_textblob'].str.strip() != ''])}, non-empty texts")
print(f"  text_vader: {len(df[df['text_vader'].str.strip() != ''])}, non-empty texts")
print(f"  text_transformer: {len(df[df['text_transformer'].str.strip() != ''])}, non-empty texts")

# Preprocessing validation test
print("=" * 60)
print("PREPROCESSING VALIDATION TEST")
print("=" * 60)

test_text = "I can't believe how AMAZING this discovery is!!! 🚀"

text_textblob = preprocess_for_textblob(test_text)
text_vader = preprocess_for_vader(test_text)
text_transformer = preprocess_for_transformer(test_text)

print(f"\nOriginal:    {test_text}")
print(f"TextBlob:    {text_textblob}")
print(f"VADER:       {text_vader}")
print(f"Transformer: {text_transformer}")

# Assertions
try:
    assert '!!!' not in text_textblob, "TextBlob: Punctuation should be removed"
    assert '🚀' not in text_textblob, "TextBlob: Emojis should be removed"
    assert text_textblob.islower(), "TextBlob: Should be lowercase"
    
    assert '!!!' in text_vader, "VADER: Punctuation MUST be preserved!"
    assert '🚀' in text_vader, "VADER: Emojis MUST be preserved!"
    assert 'AMAZING' in text_vader, "VADER: Capitalization MUST be preserved!"
    
    assert '!!!' in text_transformer, "Transformer: Punctuation MUST be preserved!"
    
    print("\n✓✓✓ All preprocessing validation tests PASSED! ✓✓✓")
except AssertionError as e:
    print(f"\n✗✗✗ VALIDATION FAILED: {e} ✗✗✗")
    raise

# Engagement feature extraction
print("\n" + "=" * 60)
print("EXTRACTING ENGAGEMENT FEATURES")
print("=" * 60)

# Vectorized features (FAST)
df['question_count'] = df['text_raw'].str.count('\?').fillna(0).astype('int32')
df['exclamation_count'] = df['text_raw'].str.count('!').fillna(0).astype('int32')
df['text_length'] = df['text_raw'].str.len().fillna(0).astype('int32')
df['word_count'] = df['text_raw'].str.split().str.len().fillna(0).astype('int32')

print("✓ Created vectorized features (question_count, exclamation_count, text_length, word_count)")

# Caps ratio (requires apply)
def calculate_caps_ratio(text):
    if pd.isna(text) or len(str(text)) == 0:
        return 0.0
    text_str = str(text)
    return sum(1 for c in text_str if c.isupper()) / len(text_str)

tqdm.pandas(desc="Calculating caps ratio")
df['caps_ratio'] = df['text_raw'].progress_apply(calculate_caps_ratio).astype('float32')
print("✓ Created caps_ratio")

# Emoji count (requires apply)
def count_emojis(text):
    if pd.isna(text):
        return 0
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        "]+", flags=re.UNICODE)
    return len(emoji_pattern.findall(str(text)))

tqdm.pandas(desc="Counting emojis")
df['emoji_count'] = df['text_raw'].progress_apply(count_emojis).astype('int32')
print("✓ Created emoji_count")

print("\n✓ All engagement features extracted")
print(f"  Question marks: {df['question_count'].sum():,} total")
print(f"  Exclamation marks: {df['exclamation_count'].sum():,} total")
print(f"  Emojis: {df['emoji_count'].sum():,} total")

# Save preprocessed data
os.makedirs('../data/processed', exist_ok=True)

output_path = '../data/processed/02_preprocessed_data.csv'
df.to_csv(output_path, index=False)

print("=" * 60)
print("SAVING PREPROCESSED DATA")
print("=" * 60)
print(f"✓ Preprocessed data saved: {output_path}")
print(f"  Columns: {len(df.columns)}")
print(f"  Rows: {len(df):,}")
print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Display column names
print("\nPreprocessing columns:")
preprocessing_cols = [col for col in df.columns if col.startswith('text_')]
for col in preprocessing_cols:
    print(f"  - {col}")

print("\nEngagement feature columns:")
engagement_cols = ['question_count', 'exclamation_count', 'text_length', 'word_count', 'caps_ratio', 'emoji_count']
for col in engagement_cols:
    if col in df.columns:
        print(f"  - {col}")

print("\n" + "=" * 60)
print("NOTEBOOK 3 COMPLETE")
print("=" * 60)
print("Next step: Run Notebook 4 (Sentiment Modeling and Validation)")

