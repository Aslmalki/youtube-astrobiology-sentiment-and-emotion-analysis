#!/usr/bin/env python3
"""
Execute Notebook 4: Sentiment Modeling and Validation
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
import warnings
from tqdm import tqdm
import gc

# Set random seed for reproducibility
np.random.seed(42)

# Configure display options
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')

print("Libraries imported successfully")

# Cell 3: Load Preprocessed Data
print("=" * 60)
print("LOADING PREPROCESSED DATA")
print("=" * 60)

df = pd.read_csv('../data/processed/02_preprocessed_data.csv')
print(f"Loaded dataset: {len(df):,} comments, {len(df.columns)} columns")

# Verify preprocessing columns exist
required_cols = ['text_textblob', 'text_vader', 'text_transformer']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Required column {col} not found in dataset")
    print(f"✓ {col} column found")

# Cell 5: TextBlob Sentiment Analysis
from textblob import TextBlob

def get_textblob_sentiment(text):
    """
    Apply TextBlob sentiment analysis.
    
    Parameters:
    -----------
    text : str
        Preprocessed text for TextBlob
        
    Returns:
    --------
    tuple : (sentiment_label, polarity_score)
    """
    if not text or pd.isna(text) or str(text).strip() == '':
        return 'neutral', 0.0
    
    try:
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            label = 'positive'
        elif polarity < -0.1:
            label = 'negative'
        else:
            label = 'neutral'
        
        return label, polarity
    except Exception as e:
        return 'neutral', 0.0

# Apply to text_textblob (heavy preprocessing)
print("=" * 60)
print("APPLYING TEXTBLOB SENTIMENT ANALYSIS")
print("=" * 60)

tqdm.pandas(desc="TextBlob")
df[['sentiment_textblob', 'polarity_textblob']] = df['text_textblob'].progress_apply(
    lambda x: pd.Series(get_textblob_sentiment(x))
)

print(f"\n✓ TextBlob complete")
print(f"\nDistribution:")
print(df['sentiment_textblob'].value_counts())
print(f"\nPolarity statistics:")
print(f"  Mean: {df['polarity_textblob'].mean():.3f}")
print(f"  Median: {df['polarity_textblob'].median():.3f}")
print(f"  Std: {df['polarity_textblob'].std():.3f}")

# Cell 7: VADER Sentiment Analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def get_vader_sentiment(text):
    """
    Apply VADER sentiment analysis.
    
    Parameters:
    -----------
    text : str
        Minimal-preprocessed text (preserves punctuation, caps, emojis)
        
    Returns:
    --------
    tuple : (sentiment_label, compound_score)
    """
    if not text or pd.isna(text) or str(text).strip() == '':
        return 'neutral', 0.0
    
    try:
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(str(text))
        compound = scores['compound']
        
        if compound >= 0.05:
            label = 'positive'
        elif compound <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        
        return label, compound
    except Exception as e:
        return 'neutral', 0.0

# Apply to text_vader (minimal preprocessing - HAS punctuation, caps, emojis!)
print("\n" + "=" * 60)
print("APPLYING VADER SENTIMENT ANALYSIS")
print("=" * 60)

tqdm.pandas(desc="VADER")
df[['sentiment_vader', 'compound_vader']] = df['text_vader'].progress_apply(
    lambda x: pd.Series(get_vader_sentiment(x))
)

print(f"\n✓ VADER complete")
print(f"\nDistribution:")
print(df['sentiment_vader'].value_counts())
print(f"\nCompound score statistics:")
print(f"  Mean: {df['compound_vader'].mean():.3f}")
print(f"  Median: {df['compound_vader'].median():.3f}")
print(f"  Std: {df['compound_vader'].std():.3f}")

# Cell 9: Transformer Sentiment Analysis
from transformers import pipeline
import torch

def get_optimal_batch_size(device):
    """Determine optimal batch size based on GPU memory."""
    if device == -1:  # CPU
        return 8
    
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory_gb >= 8:
            return 64
        elif gpu_memory_gb >= 4:
            return 32
        else:
            return 16
    return 8

def get_transformer_sentiment_batch(texts, batch_size=32):
    """
    Process sentiment analysis in batches for memory efficiency.
    
    Parameters:
    -----------
    texts : pd.Series or list
        Texts to analyze
    batch_size : int
        Batch size for processing
        
    Returns:
    --------
    list : List of sentiment results
    """
    # Check device
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")
    
    # Load model
    print("Loading transformer model...")
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device,
        truncation=True,
        max_length=512
    )
    
    results = []
    text_list = texts.tolist() if hasattr(texts, 'tolist') else list(texts)
    
    # Process in batches
    for i in tqdm(range(0, len(text_list), batch_size), desc="Transformer inference"):
        batch = text_list[i:i+batch_size]
        
        # Handle empty strings
        batch = [text if text and str(text).strip() else 'neutral text' for text in batch]
        
        try:
            batch_results = sentiment_pipeline(batch)
            results.extend(batch_results)
            
            # GPU memory management
            if device == 0 and i % (batch_size * 10) == 0:
                torch.cuda.empty_cache()
                gc.collect()
                
        except Exception as e:
            print(f"Error at batch {i}: {e}")
            results.extend([{'label': 'NEUTRAL', 'score': 0.0}] * len(batch))
    
    # Final cleanup
    if device == 0:
        torch.cuda.empty_cache()
    gc.collect()
    
    return results

# Apply to text_transformer (light preprocessing)
print("\n" + "=" * 60)
print("APPLYING TRANSFORMER SENTIMENT ANALYSIS")
print("=" * 60)

device = 0 if torch.cuda.is_available() else -1
batch_size = get_optimal_batch_size(device)
print(f"Optimal batch size: {batch_size}")

transformer_results = get_transformer_sentiment_batch(
    df['text_transformer'],
    batch_size=batch_size
)

# Extract labels and scores
df['sentiment_transformer'] = [r['label'].upper().replace('POSITIVE', 'positive').replace('NEGATIVE', 'negative').replace('NEUTRAL', 'neutral') 
                               for r in transformer_results]
df['confidence_transformer'] = [r['score'] for r in transformer_results]

print(f"\n✓ Transformer complete")
print(f"\nDistribution:")
print(df['sentiment_transformer'].value_counts())
print(f"\nConfidence statistics:")
print(f"  Mean: {df['confidence_transformer'].mean():.3f}")
print(f"  Median: {df['confidence_transformer'].median():.3f}")
print(f"  Std: {df['confidence_transformer'].std():.3f}")

# Cell 11: Save Sentiment Results
output_path = '../data/processed/03_sentiment_results.csv'
df.to_csv(output_path, index=False)

print("=" * 60)
print("SAVING SENTIMENT RESULTS")
print("=" * 60)
print(f"✓ Sentiment results saved: {output_path}")
print(f"  Rows: {len(df):,}")
print(f"  Columns: {len(df.columns)}")

# Display sentiment columns
print("\nSentiment columns:")
sentiment_cols = [col for col in df.columns if 'sentiment' in col.lower() or 'polarity' in col.lower() or 'compound' in col.lower() or 'confidence' in col.lower()]
for col in sentiment_cols:
    print(f"  - {col}")

# Summary of all models
print("\n" + "=" * 60)
print("SENTIMENT ANALYSIS SUMMARY")
print("=" * 60)

print("\nTextBlob:")
print(df['sentiment_textblob'].value_counts())
print("\nVADER:")
print(df['sentiment_vader'].value_counts())
print("\nTransformer:")
print(df['sentiment_transformer'].value_counts())

print("\n" + "=" * 60)
print("NOTEBOOK 4 COMPLETE")
print("=" * 60)
print("Next step: Run Notebook 5 (Downstream Analysis and Reporting)")

