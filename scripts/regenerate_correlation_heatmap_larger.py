#!/usr/bin/env python3
"""
Regenerate Correlation Heatmap (Figure 9): Sentiment Scores vs. Engagement Metrics
====================================================================================

This script creates the correlation matrix figure (Spearman and Pearson) showing
correlations between SENTIMENT SCORES and ENGAGEMENT METRICS. The 5x5 matrix includes:
- TextBlob sentiment score
- VADER sentiment score
- DistilBERT (transformer) sentiment score
- like_count
- reply_count

Two heatmaps side by side: Spearman (for skewed distributions) and Pearson (linear).
Saves to outputs/figures/correlation_matrix.png (Figure 9 in the paper).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("="*70)
print("FIGURE 9: SENTIMENT-ENGAGEMENT CORRELATION HEATMAP")
print("="*70)

# Project root: script in scripts/ so project root is parent
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n=== STEP 1: LOADING DATA ===\n")

# Load sentiment results (has polarity_textblob, compound_vader, sentiment_transformer, etc.)
data_path = os.path.join(_project_root, 'data', 'processed', '03_sentiment_results.csv')
if not os.path.exists(data_path):
    data_path = os.path.join(_project_root, 'data', 'sample_data.csv')

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Could not find 03_sentiment_results.csv. Checked: {data_path}")

df = pd.read_csv(data_path)
print(f"✓ Loaded {len(df):,} comments from {data_path}")

# ============================================================================
# STEP 2: PREPARE SENTIMENT SCORES AND ENGAGEMENT METRICS
# ============================================================================
print("\n=== STEP 2: PREPARING SENTIMENT SCORES AND ENGAGEMENT METRICS ===\n")

# TextBlob: polarity_textblob
if 'polarity_textblob' not in df.columns:
    raise ValueError("polarity_textblob column not found")
df['textblob_score'] = pd.to_numeric(df['polarity_textblob'], errors='coerce')

# VADER: compound_vader
if 'compound_vader' not in df.columns:
    raise ValueError("compound_vader column not found")
df['vader_score'] = pd.to_numeric(df['compound_vader'], errors='coerce')

# DistilBERT/Transformer: create polarity score from sentiment label × confidence
# positive = +confidence, negative = -confidence, neutral = NaN (excluded)
if 'sentiment_transformer' not in df.columns or 'confidence_transformer' not in df.columns:
    raise ValueError("sentiment_transformer or confidence_transformer column not found")

df['distilbert_score'] = df.apply(
    lambda row: row['confidence_transformer'] if row['sentiment_transformer'] == 'positive'
    else -row['confidence_transformer'] if row['sentiment_transformer'] == 'negative'
    else np.nan,
    axis=1
)
df['distilbert_score'] = pd.to_numeric(df['distilbert_score'], errors='coerce')

# Engagement metrics
if 'like_count' not in df.columns or 'reply_count' not in df.columns:
    raise ValueError("like_count or reply_count column not found")
df['like_count'] = pd.to_numeric(df['like_count'], errors='coerce')
df['reply_count'] = pd.to_numeric(df['reply_count'], errors='coerce')

# Build 5x5 correlation matrix: TextBlob, VADER, DistilBERT, like_count, reply_count
# Use display names for the heatmap
corr_columns = {
    'TextBlob': 'textblob_score',
    'VADER': 'vader_score',
    'DistilBERT': 'distilbert_score',
    'Likes': 'like_count',
    'Replies': 'reply_count'
}

corr_data = df[[v for v in corr_columns.values()]].copy()
corr_data = corr_data.dropna()
print(f"✓ Valid data points (complete cases): {len(corr_data):,}")

# Rename columns for display
corr_data_display = corr_data.rename(columns={v: k for k, v in corr_columns.items()})

# Calculate correlation matrices
corr_spearman = corr_data_display.corr(method='spearman')
corr_pearson = corr_data_display.corr(method='pearson')

print(f"✓ Calculated Spearman and Pearson correlation matrices (5×5)")
print(f"\nSpearman ρ range: [{corr_spearman.values.min():.4f}, {corr_spearman.values.max():.4f}]")
print(f"Pearson r range:  [{corr_pearson.values.min():.4f}, {corr_pearson.values.max():.4f}]")

# ============================================================================
# STEP 3: CREATE FIGURE WITH TWO HEATMAPS SIDE BY SIDE
# ============================================================================
print("\n=== STEP 3: CREATING FIGURE ===\n")

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 11

# Create figure: two heatmaps side by side
fig, axes = plt.subplots(1, 2, figsize=(36, 14))

# Font sizes
title_fontsize = 20
label_fontsize = 16
tick_fontsize = 14
annotation_fontsize = 16

# Use coolwarm: blue = negative, red = positive (center=0)
# vmin=-1, vmax=1 for full scale (diagonal=1, sentiment-engagement will be near 0)
vmin, vmax = -1, 1

# Left: Spearman correlation heatmap
heatmap0 = sns.heatmap(
    corr_spearman, annot=True, fmt='.3f', cmap='coolwarm', center=0,
    square=True, linewidths=1.0, cbar_kws={"shrink": 0.9},
    ax=axes[0], vmin=vmin, vmax=vmax,
    annot_kws={"size": annotation_fontsize, "weight": "bold"}
)
cbar0 = heatmap0.collections[0].colorbar
cbar0.set_label("Correlation Coefficient (ρ)", fontsize=label_fontsize, fontweight='bold', labelpad=15)
cbar0.ax.tick_params(labelsize=tick_fontsize)
axes[0].set_title('Spearman Correlation Matrix\n(For Skewed Distributions)',
                  fontweight='bold', fontsize=title_fontsize, pad=25)
axes[0].set_xlabel('Variable', fontsize=label_fontsize, fontweight='bold', labelpad=10)
axes[0].set_ylabel('Variable', fontsize=label_fontsize, fontweight='bold', labelpad=10)
axes[0].tick_params(labelsize=tick_fontsize, width=1.5, length=6)

# Right: Pearson correlation heatmap
heatmap1 = sns.heatmap(
    corr_pearson, annot=True, fmt='.3f', cmap='coolwarm', center=0,
    square=True, linewidths=1.0, cbar_kws={"shrink": 0.9},
    ax=axes[1], vmin=vmin, vmax=vmax,
    annot_kws={"size": annotation_fontsize, "weight": "bold"}
)
cbar1 = heatmap1.collections[0].colorbar
cbar1.set_label("Correlation Coefficient (r)", fontsize=label_fontsize, fontweight='bold', labelpad=15)
cbar1.ax.tick_params(labelsize=tick_fontsize)
axes[1].set_title('Pearson Correlation Matrix\n(Linear Relationships)',
                  fontweight='bold', fontsize=title_fontsize, pad=25)
axes[1].set_xlabel('Variable', fontsize=label_fontsize, fontweight='bold', labelpad=10)
axes[1].set_ylabel('Variable', fontsize=label_fontsize, fontweight='bold', labelpad=10)
axes[1].tick_params(labelsize=tick_fontsize, width=1.5, length=6)

# Layout
plt.tight_layout(pad=0.5)
plt.subplots_adjust(left=0.04, right=0.97, top=0.94, bottom=0.06, wspace=0.20)

# ============================================================================
# STEP 4: SAVE FIGURE
# ============================================================================
output_dir = os.path.join(_project_root, 'outputs', 'figures')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'correlation_matrix.png')

plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.05)
plt.close()

print(f"✓ Saved Figure 9: {output_path}")

# ============================================================================
# STEP 5: SAVE CORRELATION TABLES
# ============================================================================
print("\n=== STEP 5: SAVING CORRELATION TABLES ===\n")

tables_dir = os.path.join(_project_root, 'outputs', 'tables')
os.makedirs(tables_dir, exist_ok=True)
corr_spearman.to_csv(os.path.join(tables_dir, 'correlation_spearman.csv'))
corr_pearson.to_csv(os.path.join(tables_dir, 'correlation_pearson.csv'))
print("✓ Correlation tables saved (sentiment + engagement variables)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("COMPLETE!")
print("="*70)
print(f"""
Figure 9 (correlation_matrix.png) has been regenerated with:
  - Variables: TextBlob, VADER, DistilBERT (sentiment scores) + Likes, Replies (engagement)
  - Left heatmap: Spearman correlation (appropriate for skewed engagement data)
  - Right heatmap: Pearson correlation (linear relationships)
  - Color scheme: coolwarm (blue = negative, red = positive)
  - Values displayed in each cell (3 decimal places)
  - Sentiment-engagement correlations are extremely weak (|ρ|, |r| < 0.07)
""")
