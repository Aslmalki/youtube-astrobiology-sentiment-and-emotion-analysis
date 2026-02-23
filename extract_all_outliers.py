"""
Extract ALL Outliers to CSV for Manual Study
Creates a comprehensive CSV file with all 1,979 outliers (1.30% of dataset)
"""

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# Get project root directory (Github_Copy root when run from scripts/)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Create output directory (use outputs/ for reproducibility)
output_dir = os.path.join(project_root, 'outputs', 'outlier_analysis')
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("EXTRACTING ALL OUTLIERS FOR MANUAL STUDY")
print("=" * 80)

# Load data
print("\nLoading data...")
data_path = os.path.join(project_root, 'data', 'processed', '03_sentiment_results.csv')
if not os.path.exists(data_path):
    data_path = os.path.join(project_root, 'data', 'sample_data.csv')
df = pd.read_csv(data_path)
print(f"Loaded {len(df):,} comments")

# Ensure numeric columns are numeric
df['like_count'] = pd.to_numeric(df['like_count'], errors='coerce').fillna(0).astype('int32')
df['reply_count'] = pd.to_numeric(df['reply_count'], errors='coerce').fillna(0).astype('int32')

# Calculate 99th percentile thresholds
like_threshold = df['like_count'].quantile(0.99)
reply_threshold = df['reply_count'].quantile(0.99)

print(f"\n99th Percentile Thresholds:")
print(f"  Likes: {like_threshold:.0f}")
print(f"  Replies: {reply_threshold:.0f}")

# Identify ALL outliers: comments with likes OR replies > 99th percentile
outliers = df[(df['like_count'] > like_threshold) | (df['reply_count'] > reply_threshold)].copy()

print(f"\nTotal Outliers Identified: {len(outliers):,}")
print(f"Percentage of dataset: {len(outliers)/len(df)*100:.2f}%")

# Create text features if they don't exist
if 'text_length' not in outliers.columns and 'comment_text_original' in outliers.columns:
    outliers['text_length'] = outliers['comment_text_original'].str.len().fillna(0).astype('int32')
if 'word_count' not in outliers.columns and 'comment_text_original' in outliers.columns:
    outliers['word_count'] = outliers['comment_text_original'].str.split().str.len().fillna(0).astype('int32')
if 'question_count' not in outliers.columns and 'comment_text_original' in outliers.columns:
    outliers['question_count'] = outliers['comment_text_original'].str.count('?').fillna(0).astype('int32')
if 'exclamation_count' not in outliers.columns and 'comment_text_original' in outliers.columns:
    outliers['exclamation_count'] = outliers['comment_text_original'].str.count('!').fillna(0).astype('int32')

# Calculate combined engagement score for sorting
outliers['combined_engagement'] = (
    (outliers['like_count'] / outliers['like_count'].max()) * 0.7 +
    (outliers['reply_count'] / outliers['reply_count'].max()) * 0.3
)

# Sort by combined engagement (highest first)
outliers = outliers.sort_values('combined_engagement', ascending=False).reset_index(drop=True)

# Select relevant columns for the researcher
columns_to_include = [
    'comment_text_original',  # The actual comment text
    'like_count',
    'reply_count',
    'combined_engagement',  # For reference
    'search_query',  # Topic
    'sentiment_textblob',
    'sentiment_vader',
    'sentiment_transformer',  # All three sentiment methods
    'text_length',
    'word_count',
    'question_count',
    'exclamation_count',
    'video_view_count',  # If available
    'published_at',  # If available
    'author_name',  # If available (for context)
    'video_title',  # If available
    'channel_title'  # If available
]

# Only include columns that exist
available_columns = [col for col in columns_to_include if col in outliers.columns]
outliers_export = outliers[available_columns].copy()

# Rename columns for clarity
column_rename = {
    'comment_text_original': 'Comment_Text',
    'like_count': 'Likes',
    'reply_count': 'Replies',
    'combined_engagement': 'Combined_Engagement_Score',
    'search_query': 'Topic',
    'sentiment_textblob': 'Sentiment_TextBlob',
    'sentiment_vader': 'Sentiment_VADER',
    'sentiment_transformer': 'Sentiment_Transformer',
    'text_length': 'Text_Length_Characters',
    'word_count': 'Word_Count',
    'question_count': 'Question_Count',
    'exclamation_count': 'Exclamation_Count',
    'video_view_count': 'Video_View_Count',
    'published_at': 'Published_Date',
    'author_name': 'Author_Name',
    'video_title': 'Video_Title',
    'channel_title': 'Channel_Title'
}

# Rename columns that exist
for old_name, new_name in column_rename.items():
    if old_name in outliers_export.columns:
        outliers_export = outliers_export.rename(columns={old_name: new_name})

# Add a rank column
outliers_export.insert(0, 'Rank', range(1, len(outliers_export) + 1))

# Save to CSV
output_path = os.path.join(output_dir, 'all_outliers_complete.csv')
outliers_export.to_csv(output_path, index=False, encoding='utf-8')

print(f"\n✓ Saved ALL outliers to: {output_path}")
print(f"  Total rows: {len(outliers_export):,}")
print(f"  Total columns: {len(outliers_export.columns)}")
print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

# Print summary statistics
print("\n" + "=" * 80)
print("OUTLIER SUMMARY STATISTICS")
print("=" * 80)
print(f"\nEngagement Statistics:")
print(f"  Mean likes: {outliers['like_count'].mean():.2f}")
print(f"  Median likes: {outliers['like_count'].median():.2f}")
print(f"  Max likes: {outliers['like_count'].max():,}")
print(f"  Mean replies: {outliers['reply_count'].mean():.2f}")
print(f"  Median replies: {outliers['reply_count'].median():.2f}")
print(f"  Max replies: {outliers['reply_count'].max():,}")

if 'search_query' in outliers.columns:
    print(f"\nTopic Distribution:")
    topic_counts = outliers['search_query'].value_counts()
    for topic, count in topic_counts.items():
        pct = (count / len(outliers)) * 100
        print(f"  {topic}: {count:,} ({pct:.1f}%)")

if 'sentiment_transformer' in outliers.columns:
    print(f"\nSentiment Distribution (Transformer):")
    sentiment_counts = outliers['sentiment_transformer'].value_counts()
    for sentiment, count in sentiment_counts.items():
        pct = (count / len(outliers)) * 100
        print(f"  {sentiment}: {count:,} ({pct:.1f}%)")

print(f"\nText Characteristics:")
print(f"  Mean text length: {outliers['text_length'].mean():.1f} characters")
print(f"  Mean word count: {outliers['word_count'].mean():.1f} words")
print(f"  Mean questions: {outliers['question_count'].mean():.2f}")
print(f"  Mean exclamations: {outliers['exclamation_count'].mean():.2f}")

print("\n" + "=" * 80)
print("EXTRACTION COMPLETE")
print("=" * 80)
print(f"\nThe CSV file contains ALL {len(outliers_export):,} outliers with the following information:")
print("  - Comment text (full)")
print("  - Engagement metrics (likes, replies)")
print("  - Topic")
print("  - Sentiment (all three methods)")
print("  - Text characteristics (length, word count, questions, exclamations)")
print("  - Video/channel metadata (if available)")
print("  - Sorted by combined engagement score (highest first)")
print("\nThis file is ready for manual study by researchers.")



