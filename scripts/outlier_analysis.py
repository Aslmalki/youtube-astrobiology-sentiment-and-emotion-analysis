"""
Comprehensive Outlier Analysis for YouTube Comments Dataset
Identifies high-engagement outliers and extracts examples for supervisor review
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Get project root directory (Github_Copy root when run from scripts/)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Create output directory (use outputs/ for reproducibility)
output_dir = os.path.join(project_root, 'outputs', 'outlier_analysis')
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("COMPREHENSIVE OUTLIER ANALYSIS")
print("=" * 80)

# Load data
print("\nLoading data...")
data_path = os.path.join(project_root, 'data', 'processed', '03_sentiment_results.csv')
if not os.path.exists(data_path):
    data_path = os.path.join(project_root, 'data', 'sample_data.csv')
df = pd.read_csv(data_path)
print(f"Loaded {len(df):,} comments")

# Ensure numeric columns are numeric
numeric_cols = ['like_count', 'reply_count', 'text_length', 'word_count']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Create text features if they don't exist
if 'text_length' not in df.columns and 'comment_text_original' in df.columns:
    df['text_length'] = df['comment_text_original'].str.len().fillna(0).astype('int32')
if 'word_count' not in df.columns and 'comment_text_original' in df.columns:
    df['word_count'] = df['comment_text_original'].str.split().str.len().fillna(0).astype('int32')
if 'question_count' not in df.columns and 'comment_text_original' in df.columns:
    df['question_count'] = df['comment_text_original'].str.count('?').fillna(0).astype('int32')
if 'exclamation_count' not in df.columns and 'comment_text_original' in df.columns:
    df['exclamation_count'] = df['comment_text_original'].str.count('!').fillna(0).astype('int32')

print("\n" + "=" * 80)
print("TASK 1: IDENTIFY OUTLIERS")
print("=" * 80)

# Calculate 99th percentile thresholds
like_threshold = df['like_count'].quantile(0.99)
reply_threshold = df['reply_count'].quantile(0.99)

print(f"\n99th Percentile Thresholds:")
print(f"  Likes: {like_threshold:.0f}")
print(f"  Replies: {reply_threshold:.0f}")

# Identify outliers: comments with likes OR replies > 99th percentile
outliers = df[(df['like_count'] > like_threshold) | (df['reply_count'] > reply_threshold)].copy()
typical = df[~((df['like_count'] > like_threshold) | (df['reply_count'] > reply_threshold))].copy()

print(f"\nOutlier Statistics:")
print(f"  Total outliers: {len(outliers):,}")
print(f"  Percentage of dataset: {len(outliers)/len(df)*100:.2f}%")
print(f"  Outliers by likes only: {len(outliers[outliers['like_count'] > like_threshold]) - len(outliers[(outliers['like_count'] > like_threshold) & (outliers['reply_count'] > reply_threshold)]):,}")
print(f"  Outliers by replies only: {len(outliers[outliers['reply_count'] > reply_threshold]) - len(outliers[(outliers['like_count'] > like_threshold) & (outliers['reply_count'] > reply_threshold)]):,}")
print(f"  Outliers by both: {len(outliers[(outliers['like_count'] > like_threshold) & (outliers['reply_count'] > reply_threshold)]):,}")

print(f"\n  Mean likes (outliers): {outliers['like_count'].mean():.2f}")
print(f"  Mean replies (outliers): {outliers['reply_count'].mean():.2f}")
print(f"  Max likes: {outliers['like_count'].max():,}")
print(f"  Max replies: {outliers['reply_count'].max():,}")

print("\n" + "=" * 80)
print("TASK 2: COMPARISON TABLE")
print("=" * 80)

# Prepare comparison data
comparison_data = []

# Engagement metrics
comparison_data.append({
    'Metric': 'Mean Likes',
    'Outliers': outliers['like_count'].mean(),
    'Typical': typical['like_count'].mean(),
    'Difference': outliers['like_count'].mean() - typical['like_count'].mean()
})

comparison_data.append({
    'Metric': 'Median Likes',
    'Outliers': outliers['like_count'].median(),
    'Typical': typical['like_count'].median(),
    'Difference': outliers['like_count'].median() - typical['like_count'].median()
})

comparison_data.append({
    'Metric': 'Mean Replies',
    'Outliers': outliers['reply_count'].mean(),
    'Typical': typical['reply_count'].mean(),
    'Difference': outliers['reply_count'].mean() - typical['reply_count'].mean()
})

comparison_data.append({
    'Metric': 'Median Replies',
    'Outliers': outliers['reply_count'].median(),
    'Typical': typical['reply_count'].median(),
    'Difference': outliers['reply_count'].median() - typical['reply_count'].median()
})

# Text features
comparison_data.append({
    'Metric': 'Mean Text Length',
    'Outliers': outliers['text_length'].mean(),
    'Typical': typical['text_length'].mean(),
    'Difference': outliers['text_length'].mean() - typical['text_length'].mean()
})

comparison_data.append({
    'Metric': 'Mean Word Count',
    'Outliers': outliers['word_count'].mean(),
    'Typical': typical['word_count'].mean(),
    'Difference': outliers['word_count'].mean() - typical['word_count'].mean()
})

comparison_data.append({
    'Metric': 'Mean Question Count',
    'Outliers': outliers['question_count'].mean(),
    'Typical': typical['question_count'].mean(),
    'Difference': outliers['question_count'].mean() - typical['question_count'].mean()
})

comparison_data.append({
    'Metric': 'Mean Exclamation Count',
    'Outliers': outliers['exclamation_count'].mean(),
    'Typical': typical['exclamation_count'].mean(),
    'Difference': outliers['exclamation_count'].mean() - typical['exclamation_count'].mean()
})

# Sentiment percentages (using transformer model as primary)
if 'sentiment_transformer' in df.columns:
    for sentiment in ['positive', 'negative', 'neutral']:
        outlier_pct = (outliers['sentiment_transformer'] == sentiment).sum() / len(outliers) * 100
        typical_pct = (typical['sentiment_transformer'] == sentiment).sum() / len(typical) * 100
        comparison_data.append({
            'Metric': f'Sentiment: {sentiment.capitalize()} (%)',
            'Outliers': outlier_pct,
            'Typical': typical_pct,
            'Difference': outlier_pct - typical_pct
        })

# Topic percentages
if 'search_query' in df.columns:
    topics = df['search_query'].unique()
    for topic in topics:
        outlier_pct = (outliers['search_query'] == topic).sum() / len(outliers) * 100
        typical_pct = (typical['search_query'] == topic).sum() / len(typical) * 100
        comparison_data.append({
            'Metric': f'Topic: {topic} (%)',
            'Outliers': outlier_pct,
            'Typical': typical_pct,
            'Difference': outlier_pct - typical_pct
        })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.round(2)

# Save comparison table
comparison_path = os.path.join(output_dir, 'outlier_comparison_table.csv')
comparison_df.to_csv(comparison_path, index=False)
print(f"\n✓ Comparison table saved: {comparison_path}")
print("\nComparison Table Preview:")
print(comparison_df.head(15).to_string(index=False))

print("\n" + "=" * 80)
print("TASK 3: EXTRACT TOP EXAMPLES")
print("=" * 80)

# Calculate combined engagement score (normalized)
outliers['combined_engagement'] = (
    (outliers['like_count'] / outliers['like_count'].max()) * 0.7 +
    (outliers['reply_count'] / outliers['reply_count'].max()) * 0.3
)

# Top 5 by likes
top_likes = outliers.nlargest(5, 'like_count')[
    ['comment_text_original', 'like_count', 'reply_count', 'search_query', 
     'sentiment_transformer', 'text_length', 'question_count', 'exclamation_count']
].copy()
top_likes['rank_type'] = 'Top by Likes'
top_likes['rank'] = range(1, 6)

# Top 5 by replies
top_replies = outliers.nlargest(5, 'reply_count')[
    ['comment_text_original', 'like_count', 'reply_count', 'search_query',
     'sentiment_transformer', 'text_length', 'question_count', 'exclamation_count']
].copy()
top_replies['rank_type'] = 'Top by Replies'
top_replies['rank'] = range(1, 6)

# Top 5 by combined engagement
top_combined = outliers.nlargest(5, 'combined_engagement')[
    ['comment_text_original', 'like_count', 'reply_count', 'search_query',
     'sentiment_transformer', 'text_length', 'question_count', 'exclamation_count']
].copy()
top_combined['rank_type'] = 'Top by Combined Engagement'
top_combined['rank'] = range(1, 6)

# Combine all top examples
top_examples = pd.concat([top_likes, top_replies, top_combined], ignore_index=True)

# Rename columns for clarity
top_examples.columns = ['Comment Text', 'Likes', 'Replies', 'Topic', 'Sentiment',
                        'Text Length', 'Question Count', 'Exclamation Count', 
                        'Rank Type', 'Rank']

# Save top examples
examples_path = os.path.join(output_dir, 'top_outlier_examples.csv')
top_examples.to_csv(examples_path, index=False)
print(f"\n✓ Top examples saved: {examples_path}")
print("\nTop 5 by Likes:")
print(top_likes[['comment_text_original', 'like_count', 'reply_count', 'search_query', 'sentiment_transformer']].to_string(index=False))

print("\n" + "=" * 80)
print("TASK 4: TOPIC ANALYSIS")
print("=" * 80)

if 'search_query' in df.columns:
    topic_analysis = []
    
    for topic in df['search_query'].unique():
        topic_outliers = outliers[outliers['search_query'] == topic]
        topic_all = df[df['search_query'] == topic]
        
        # Most common sentiment in outliers
        if len(topic_outliers) > 0 and 'sentiment_transformer' in topic_outliers.columns:
            common_sentiment = topic_outliers['sentiment_transformer'].mode()
            common_sentiment_str = common_sentiment.iloc[0] if len(common_sentiment) > 0 else 'N/A'
            sentiment_pct = (topic_outliers['sentiment_transformer'] == common_sentiment_str).sum() / len(topic_outliers) * 100
        else:
            common_sentiment_str = 'N/A'
            sentiment_pct = 0
        
        topic_analysis.append({
            'Topic': topic,
            'Number of Outliers': len(topic_outliers),
            'Percentage of Outliers': len(topic_outliers) / len(topic_all) * 100 if len(topic_all) > 0 else 0,
            'Mean Likes (Outliers)': topic_outliers['like_count'].mean() if len(topic_outliers) > 0 else 0,
            'Mean Replies (Outliers)': topic_outliers['reply_count'].mean() if len(topic_outliers) > 0 else 0,
            'Mean Combined Engagement': topic_outliers['combined_engagement'].mean() if len(topic_outliers) > 0 else 0,
            'Most Common Sentiment': common_sentiment_str,
            'Sentiment Percentage': sentiment_pct
        })
    
    topic_analysis_df = pd.DataFrame(topic_analysis)
    topic_analysis_df = topic_analysis_df.round(2)
    
    # Save topic analysis
    topic_path = os.path.join(output_dir, 'outlier_topic_analysis.csv')
    topic_analysis_df.to_csv(topic_path, index=False)
    print(f"\n✓ Topic analysis saved: {topic_path}")
    print("\nTopic Analysis:")
    print(topic_analysis_df.to_string(index=False))
else:
    print("WARNING: 'search_query' column not found. Skipping topic analysis.")

print("\n" + "=" * 80)
print("TASK 5: CATEGORIZE OUTLIERS")
print("=" * 80)

# Get top 20 outliers by combined engagement
top_20 = outliers.nlargest(20, 'combined_engagement').copy()

def categorize_comment(text):
    """Categorize comment based on keywords and patterns"""
    if pd.isna(text) or text == '':
        return 'Other'
    
    text_lower = str(text).lower()
    
    # Humorous
    humor_keywords = ['lol', 'haha', 'funny', 'joke', 'hilarious', '😂', '😄', '😆']
    if any(keyword in text_lower for keyword in humor_keywords):
        return 'Humorous'
    
    # Question
    if '?' in str(text) or any(q in text_lower for q in ['what', 'why', 'how', 'when', 'where', 'who']):
        if text_lower.count('?') >= 2 or len([q for q in ['what', 'why', 'how', 'when', 'where', 'who'] if q in text_lower]) >= 2:
            return 'Question'
    
    # Enthusiastic
    enthusiasm_keywords = ['amazing', 'incredible', 'wow', 'awesome', 'fantastic', 'love', '❤️', '🔥', '!!!']
    if any(keyword in text_lower for keyword in enthusiasm_keywords) or text.count('!') >= 3:
        return 'Enthusiastic'
    
    # Skeptical
    skeptical_keywords = ['doubt', 'skeptical', 'unlikely', 'probably not', 'doubtful', 'questionable', 'suspicious']
    if any(keyword in text_lower for keyword in skeptical_keywords):
        return 'Skeptical'
    
    # Insightful (longer comments with technical terms)
    if len(text) > 200:
        technical_terms = ['theory', 'hypothesis', 'evidence', 'research', 'study', 'data', 'analysis', 'scientific']
        if any(term in text_lower for term in technical_terms):
            return 'Insightful'
    
    # Meta-commentary
    meta_keywords = ['comment', 'video', 'channel', 'subscribe', 'like this', 'agree', 'disagree', 'opinion']
    if any(keyword in text_lower for keyword in meta_keywords):
        return 'Meta-commentary'
    
    return 'Other'

# Apply categorization
top_20['Category'] = top_20['comment_text_original'].apply(categorize_comment)

# Select columns for output
categorized = top_20[[
    'comment_text_original', 'like_count', 'reply_count', 'search_query',
    'sentiment_transformer', 'text_length', 'question_count', 'exclamation_count',
    'Category', 'combined_engagement'
]].copy()

categorized.columns = [
    'Comment Text', 'Likes', 'Replies', 'Topic', 'Sentiment',
    'Text Length', 'Question Count', 'Exclamation Count', 'Category', 'Combined Engagement'
]

# Sort by combined engagement
categorized = categorized.sort_values('Combined Engagement', ascending=False).reset_index(drop=True)

# Save categorized outliers
categorized_path = os.path.join(output_dir, 'categorized_outliers.csv')
categorized.to_csv(categorized_path, index=False)
print(f"\n✓ Categorized outliers saved: {categorized_path}")
print("\nCategory Distribution:")
print(categorized['Category'].value_counts())
print("\nTop 5 Categorized Examples:")
print(categorized.head(5)[['Comment Text', 'Likes', 'Replies', 'Category']].to_string(index=False))

print("\n" + "=" * 80)
print("TASK 6: VISUALIZATIONS")
print("=" * 80)

# 1. Scatter plot: likes vs replies (outliers highlighted)
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(typical['like_count'], typical['reply_count'], alpha=0.3, s=10, 
           color='lightblue', label='Typical Comments')
ax.scatter(outliers['like_count'], outliers['reply_count'], alpha=0.7, s=50,
           color='red', edgecolors='black', linewidth=0.5, label='Outliers')
ax.set_xlabel('Like Count', fontweight='bold', fontsize=12)
ax.set_ylabel('Reply Count', fontweight='bold', fontsize=12)
ax.set_title('Engagement Scatter Plot: Likes vs Replies\n(Outliers Highlighted)', 
             fontweight='bold', fontsize=14)
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
ax.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
scatter_path = os.path.join(output_dir, 'scatter_likes_vs_replies.png')
plt.savefig(scatter_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Saved: {scatter_path}")
plt.close()

# 2. Box plot: engagement by topic
if 'search_query' in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Likes by topic
    topics_list = df['search_query'].unique()
    data_likes = [df[df['search_query'] == topic]['like_count'].values for topic in topics_list]
    bp1 = axes[0].boxplot(data_likes, labels=topics_list, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.7)
    axes[0].set_ylabel('Like Count', fontweight='bold', fontsize=12)
    axes[0].set_title('Like Count Distribution by Topic', fontweight='bold', fontsize=13)
    axes[0].set_yscale('log')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(alpha=0.3, linestyle='--', axis='y')
    
    # Replies by topic
    data_replies = [df[df['search_query'] == topic]['reply_count'].values for topic in topics_list]
    bp2 = axes[1].boxplot(data_replies, labels=topics_list, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('orange')
        patch.set_alpha(0.7)
    axes[1].set_ylabel('Reply Count', fontweight='bold', fontsize=12)
    axes[1].set_title('Reply Count Distribution by Topic', fontweight='bold', fontsize=13)
    axes[1].set_yscale('log')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    boxplot_path = os.path.join(output_dir, 'boxplot_engagement_by_topic.png')
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {boxplot_path}")
    plt.close()

# 3. Bar chart: outliers per topic
if 'search_query' in df.columns:
    outlier_counts = outliers['search_query'].value_counts().sort_index()
    outlier_pcts = (outlier_counts / df['search_query'].value_counts().sort_index() * 100).sort_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Count
    axes[0].bar(range(len(outlier_counts)), outlier_counts.values, color='steelblue', edgecolor='black')
    axes[0].set_xticks(range(len(outlier_counts)))
    axes[0].set_xticklabels(outlier_counts.index, rotation=45, ha='right')
    axes[0].set_ylabel('Number of Outliers', fontweight='bold', fontsize=12)
    axes[0].set_title('Number of Outliers by Topic', fontweight='bold', fontsize=13)
    axes[0].grid(alpha=0.3, linestyle='--', axis='y')
    
    # Add value labels
    for i, v in enumerate(outlier_counts.values):
        axes[0].text(i, v, str(int(v)), ha='center', va='bottom', fontweight='bold')
    
    # Percentage
    axes[1].bar(range(len(outlier_pcts)), outlier_pcts.values, color='orange', edgecolor='black')
    axes[1].set_xticks(range(len(outlier_pcts)))
    axes[1].set_xticklabels(outlier_pcts.index, rotation=45, ha='right')
    axes[1].set_ylabel('Percentage of Outliers (%)', fontweight='bold', fontsize=12)
    axes[1].set_title('Percentage of Outliers by Topic', fontweight='bold', fontsize=13)
    axes[1].grid(alpha=0.3, linestyle='--', axis='y')
    
    # Add value labels
    for i, v in enumerate(outlier_pcts.values):
        axes[1].text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    bar_topic_path = os.path.join(output_dir, 'bar_outliers_per_topic.png')
    plt.savefig(bar_topic_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {bar_topic_path}")
    plt.close()

# 4. Bar chart: outlier categories
if 'Category' in top_20.columns:
    category_counts = top_20['Category'].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(category_counts)), category_counts.values, 
                   color='steelblue', edgecolor='black')
    ax.set_xticks(range(len(category_counts)))
    ax.set_xticklabels(category_counts.index, rotation=45, ha='right')
    ax.set_ylabel('Number of Outliers', fontweight='bold', fontsize=12)
    ax.set_title('Distribution of Outlier Categories\n(Top 20 Outliers)', fontweight='bold', fontsize=13)
    ax.grid(alpha=0.3, linestyle='--', axis='y')
    
    # Add value labels
    for i, v in enumerate(category_counts.values):
        ax.text(i, v, str(int(v)), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    bar_cat_path = os.path.join(output_dir, 'bar_outlier_categories.png')
    plt.savefig(bar_cat_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {bar_cat_path}")
    plt.close()

print("\n" + "=" * 80)
print("TASK 7: SUMMARY STATISTICS")
print("=" * 80)

# Generate formatted summary sentences
summary_sentences = []

summary_sentences.append(f"We identified {len(outliers):,} outliers ({len(outliers)/len(df)*100:.2f}% of the dataset) defined as comments with likes or replies exceeding the 99th percentile threshold (likes > {like_threshold:.0f} or replies > {reply_threshold:.0f}).")

summary_sentences.append(f"Outliers exhibited substantially higher engagement than typical comments, with mean likes of {outliers['like_count'].mean():.2f} (vs. {typical['like_count'].mean():.2f} for typical comments) and mean replies of {outliers['reply_count'].mean():.2f} (vs. {typical['reply_count'].mean():.2f} for typical comments).")

summary_sentences.append(f"The most extreme outlier received {outliers['like_count'].max():,} likes and {outliers['reply_count'].max():,} replies, demonstrating the highly skewed nature of engagement distribution.")

if 'search_query' in df.columns:
    topic_with_most = outliers['search_query'].value_counts().index[0]
    topic_count = outliers['search_query'].value_counts().iloc[0]
    topic_pct = (outliers['search_query'].value_counts().iloc[0] / len(outliers)) * 100
    summary_sentences.append(f"Outliers were distributed across all topics, with '{topic_with_most}' containing the highest number of outliers ({topic_count}, {topic_pct:.1f}% of all outliers).")

if 'sentiment_transformer' in outliers.columns:
    common_sentiment = outliers['sentiment_transformer'].mode().iloc[0] if len(outliers['sentiment_transformer'].mode()) > 0 else 'N/A'
    sentiment_pct = (outliers['sentiment_transformer'] == common_sentiment).sum() / len(outliers) * 100
    summary_sentences.append(f"The most common sentiment among outliers was {common_sentiment} ({sentiment_pct:.1f}% of outliers), compared to {typical['sentiment_transformer'].mode().iloc[0] if len(typical['sentiment_transformer'].mode()) > 0 else 'N/A'} in typical comments.")

summary_sentences.append(f"Outliers had an average text length of {outliers['text_length'].mean():.1f} characters (vs. {typical['text_length'].mean():.1f} for typical comments), suggesting that longer comments may be more likely to generate high engagement.")

if 'Category' in top_20.columns:
    most_common_cat = top_20['Category'].value_counts().index[0]
    cat_count = top_20['Category'].value_counts().iloc[0]
    # Build the "followed by" part separately to avoid f-string backslash issue
    follow_parts = []
    for cat, count in top_20['Category'].value_counts().head(3).iloc[1:].items():
        follow_parts.append(f"'{cat}' ({count})")
    follow_str = ', '.join(follow_parts) if follow_parts else 'others'
    summary_sentences.append(f"Among the top 20 outliers, the most common category was '{most_common_cat}' ({cat_count} comments), followed by {follow_str}.")

# Print summary sentences
print("\n" + "=" * 80)
print("SUMMARY STATISTICS (Ready for Paper)")
print("=" * 80)
print("\nCopy these sentences into your paper:\n")
for i, sentence in enumerate(summary_sentences, 1):
    print(f"{i}. {sentence}\n")

# Save summary to file
summary_path = os.path.join(output_dir, 'summary_statistics.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("OUTLIER ANALYSIS SUMMARY STATISTICS\n")
    f.write("=" * 80 + "\n\n")
    f.write("Copy these sentences into your paper:\n\n")
    for i, sentence in enumerate(summary_sentences, 1):
        f.write(f"{i}. {sentence}\n\n")
print(f"\n✓ Summary statistics saved: {summary_path}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nAll outputs saved to: {output_dir}/")
print("\nGenerated Files:")
print("  CSV Files:")
print("    - outlier_comparison_table.csv")
print("    - top_outlier_examples.csv")
print("    - outlier_topic_analysis.csv")
print("    - categorized_outliers.csv")
print("  PNG Files:")
print("    - scatter_likes_vs_replies.png")
print("    - boxplot_engagement_by_topic.png")
print("    - bar_outliers_per_topic.png")
print("    - bar_outlier_categories.png")
print("  Text Files:")
print("    - summary_statistics.txt")

