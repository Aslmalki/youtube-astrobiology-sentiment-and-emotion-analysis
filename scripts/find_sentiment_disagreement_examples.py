"""
Find Examples of Sentiment Disagreement Across Methods
========================================================

Creates a CSV file with 3 examples showing how the same comment text
gets different sentiment classifications from TextBlob, VADER, and Transformer.
"""

import pandas as pd
import numpy as np
import html

print("=" * 70)
print("FINDING SENTIMENT DISAGREEMENT EXAMPLES")
print("=" * 70)
print()

# Load data
print("=== STEP 1: LOADING DATA ===")
print()

data_file = 'data/processed/03_sentiment_results.csv'
print(f"Loading {data_file}...")

# Load only necessary columns to save memory
cols_to_load = [
    'comment_text',
    'sentiment_textblob',
    'polarity_textblob',
    'sentiment_vader',
    'compound_vader',
    'sentiment_transformer',
    'confidence_transformer'
]

df = pd.read_csv(data_file, usecols=cols_to_load)

print(f"✓ Loaded {len(df):,} comments")
print()

# ============================================================================
# STEP 2: FIND DISAGREEMENTS
# ============================================================================
print("=== STEP 2: FINDING DISAGREEMENTS ===")
print()

# Filter out rows with missing sentiment data
df_clean = df.dropna(subset=['sentiment_textblob', 'sentiment_vader', 'sentiment_transformer'])

# Find cases where all three methods disagree
all_disagree = (
    (df_clean['sentiment_textblob'] != df_clean['sentiment_vader']) &
    (df_clean['sentiment_textblob'] != df_clean['sentiment_transformer']) &
    (df_clean['sentiment_vader'] != df_clean['sentiment_transformer'])
)

# Find cases where at least two methods disagree
two_disagree = (
    (df_clean['sentiment_textblob'] != df_clean['sentiment_vader']) |
    (df_clean['sentiment_textblob'] != df_clean['sentiment_transformer']) |
    (df_clean['sentiment_vader'] != df_clean['sentiment_transformer'])
)

print(f"Comments where all 3 methods disagree: {all_disagree.sum():,}")
print(f"Comments where at least 2 methods disagree: {two_disagree.sum():,}")
print()

# ============================================================================
# STEP 3: SELECT BEST EXAMPLES
# ============================================================================
print("=== STEP 3: SELECTING BEST EXAMPLES ===")
print()

# Prioritize examples where all three disagree
candidates_all_disagree = df_clean[all_disagree].copy()

# If we don't have enough with all three disagreeing, supplement with two-disagree cases
if len(candidates_all_disagree) < 3:
    candidates_two_disagree = df_clean[two_disagree & ~all_disagree].copy()
    candidates = pd.concat([candidates_all_disagree, candidates_two_disagree]).head(10)
else:
    candidates = candidates_all_disagree.head(10)

# Filter for comments with reasonable length (not too short, not too long)
# Also filter out HTML entities and very technical text for better readability
candidates = candidates[
    (candidates['comment_text'].str.len() >= 30) &
    (candidates['comment_text'].str.len() <= 300) &
    (~candidates['comment_text'].str.contains('&quot;|&amp;|&lt;|&gt;', regex=True))  # Avoid HTML entities
].copy()

# Select 3 diverse examples
# Try to get one of each: all positive/negative/neutral mix
selected_indices = []

# Strategy: Get examples with different disagreement patterns
patterns = [
    # Pattern 1: TextBlob=positive, VADER=negative, Transformer=neutral
    ((candidates['sentiment_textblob'] == 'positive') &
     (candidates['sentiment_vader'] == 'negative') &
     (candidates['sentiment_transformer'] == 'neutral')),
    
    # Pattern 2: TextBlob=negative, VADER=positive, Transformer=neutral
    ((candidates['sentiment_textblob'] == 'negative') &
     (candidates['sentiment_vader'] == 'positive') &
     (candidates['sentiment_transformer'] == 'neutral')),
    
    # Pattern 3: TextBlob=neutral, VADER=positive, Transformer=negative
    ((candidates['sentiment_textblob'] == 'neutral') &
     (candidates['sentiment_vader'] == 'positive') &
     (candidates['sentiment_transformer'] == 'negative')),
]

# Try to get one example for each pattern
for pattern in patterns:
    matches = candidates[pattern]
    if len(matches) > 0:
        selected_indices.append(matches.index[0])
        if len(selected_indices) >= 3:
            break

# If we don't have 3 yet, fill with any remaining disagreements
if len(selected_indices) < 3:
    remaining = candidates[~candidates.index.isin(selected_indices)]
    needed = 3 - len(selected_indices)
    selected_indices.extend(remaining.head(needed).index.tolist())

# Get the selected examples
examples_df = candidates.loc[selected_indices[:3]].copy()

# ============================================================================
# STEP 4: FORMAT OUTPUT
# ============================================================================
print("=== STEP 4: FORMATTING OUTPUT ===")
print()

# Create a clean output DataFrame
output_data = []

for idx, row in examples_df.iterrows():
    # Clean HTML entities from comment text
    comment_text = html.unescape(str(row['comment_text']))
    
    output_data.append({
        'Example_Number': len(output_data) + 1,
        'Comment_Text': comment_text,
        'TextBlob_Label': row['sentiment_textblob'],
        'TextBlob_Score': f"{row['polarity_textblob']:.3f}",
        'VADER_Label': row['sentiment_vader'],
        'VADER_Score': f"{row['compound_vader']:.3f}",
        'Transformer_Label': row['sentiment_transformer'],
        'Transformer_Confidence': f"{row['confidence_transformer']:.3f}",
        'Disagreement_Type': 'All 3 methods disagree' if idx in all_disagree.index and all_disagree.loc[idx] else '2 methods disagree'
    })

output_df = pd.DataFrame(output_data)

# ============================================================================
# STEP 5: SAVE RESULTS
# ============================================================================
print("=== STEP 5: SAVING RESULTS ===")
print()

output_path = 'outputs/tables/sentiment_method_disagreement_examples.csv'
output_df.to_csv(output_path, index=False)

print(f"✓ Saved examples to: {output_path}")
print()

# Display the examples
print("=" * 70)
print("SELECTED EXAMPLES")
print("=" * 70)
print()

for idx, row in output_df.iterrows():
    print(f"Example {row['Example_Number']}:")
    print(f"  Comment: {row['Comment_Text'][:100]}..." if len(row['Comment_Text']) > 100 else f"  Comment: {row['Comment_Text']}")
    print(f"  TextBlob: {row['TextBlob_Label']} (score: {row['TextBlob_Score']})")
    print(f"  VADER: {row['VADER_Label']} (score: {row['VADER_Score']})")
    print(f"  Transformer: {row['Transformer_Label']} (confidence: {row['Transformer_Confidence']})")
    print(f"  Disagreement: {row['Disagreement_Type']}")
    print()

print("=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)

