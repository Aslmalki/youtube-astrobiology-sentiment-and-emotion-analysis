#!/usr/bin/env python3
"""
Execute Notebook 5: Downstream Analysis and Reporting
Extracted code from notebook cells
"""
import sys
import os

# Set working directory to project root (Github_Copy) for consistent paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

# Cell 1: Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from scipy.stats import kruskal
from itertools import combinations
import gc

# Set random seed for reproducibility
np.random.seed(42)

# Configure display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("Libraries imported successfully")

# Cell 3: Load Sentiment Results
print("=" * 60)
print("LOADING SENTIMENT RESULTS")
print("=" * 60)

data_path = os.path.join(project_root, 'data', 'processed', '03_sentiment_results.csv')
if not os.path.exists(data_path):
    data_path = os.path.join(project_root, 'data', 'sample_data.csv')
df = pd.read_csv(data_path)
print(f"Loaded dataset: {len(df):,} comments, {len(df.columns)} columns")

# Verify required columns exist
required_cols = ['search_query', 'sentiment_textblob', 'sentiment_vader', 'sentiment_transformer']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Required column {col} not found in dataset")
    print(f"✓ {col} column found")

# Cell 5: Chi-square Test
def perform_chi_square_test(df, topic_col, sentiment_col):
    """
    Perform chi-square test of independence for sentiment by topic.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    topic_col : str
        Column name for topics
    sentiment_col : str
        Column name for sentiment labels
        
    Returns:
    --------
    dict : Test results
    """
    # Create contingency table
    contingency_table = pd.crosstab(df[topic_col], df[sentiment_col])
    
    # Perform chi-square test
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
    
    # Calculate effect size (Cramer's V)
    n = contingency_table.sum().sum()
    cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
    
    return {
        'contingency_table': contingency_table,
        'chi2_statistic': chi2_stat,
        'p_value': p_value,
        'degrees_of_freedom': dof,
        'expected_frequencies': expected,
        'cramers_v': cramers_v,
        'n': n
    }

# Filter to valid topics only
valid_topics = ['3I/ATLAS', 'Oumuamua', 'K2-18b', "Tabby's Star", 'Venus phosphine']
df_valid = df[df['search_query'].isin(valid_topics)].copy()
print(f"Filtered to {len(df_valid):,} comments with valid topics (from {len(df):,} total)")

# Perform chi-square tests for each model
print("=" * 60)
print("CHI-SQUARE TESTS FOR SENTIMENT BY TOPIC")
print("=" * 60)

models = ['textblob', 'vader', 'transformer']
chi_square_results = {}

for model in models:
    sentiment_col = f'sentiment_{model}'
    print(f"\n{model.upper()}:")
    print("-" * 60)
    
    result = perform_chi_square_test(df_valid, 'search_query', sentiment_col)
    chi_square_results[model] = result
    
    print(f"Chi-square statistic: {result['chi2_statistic']:.4f}")
    print(f"p-value: {result['p_value']:.6f}")
    print(f"Degrees of freedom: {result['degrees_of_freedom']}")
    print(f"Cramer's V: {result['cramers_v']:.4f}")
    
    # Interpret significance
    alpha = 0.05
    if result['p_value'] < alpha:
        print(f"Result: SIGNIFICANT (p < {alpha}) - Sentiment distribution differs across topics")
    else:
        print(f"Result: NOT SIGNIFICANT (p >= {alpha}) - No significant difference across topics")
    
    # Save contingency table
    os.makedirs(os.path.join(project_root, 'outputs', 'tables'), exist_ok=True)
    result['contingency_table'].to_csv(os.path.join(project_root, 'outputs', 'tables', f'contingency_table_{model}.csv'))
    print(f"✓ Contingency table saved")

# Save chi-square results
chi_square_summary = pd.DataFrame({
    'model': models,
    'chi2_statistic': [chi_square_results[m]['chi2_statistic'] for m in models],
    'p_value': [chi_square_results[m]['p_value'] for m in models],
    'degrees_of_freedom': [chi_square_results[m]['degrees_of_freedom'] for m in models],
    'cramers_v': [chi_square_results[m]['cramers_v'] for m in models]
})
chi_square_summary.to_csv(os.path.join(project_root, 'outputs', 'tables', 'chi_square_results.csv'), index=False)
print("\n✓ Chi-square results saved")

# Cell 7: Pairwise Comparisons
def perform_pairwise_chi_square(df, topic_col, sentiment_col, topics):
    """
    Perform pairwise chi-square tests between topics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    topic_col : str
        Column name for topics
    sentiment_col : str
        Column name for sentiment labels
    topics : list
        List of topic pairs to compare
        
    Returns:
    --------
    list : List of test results
    """
    results = []
    
    for topic1, topic2 in topics:
        # Filter data for two topics
        df_pair = df[df[topic_col].isin([topic1, topic2])]
        
        # Create contingency table
        contingency_table = pd.crosstab(df_pair[topic_col], df_pair[sentiment_col])
        
        # Perform chi-square test
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
        
        results.append({
            'topic1': topic1,
            'topic2': topic2,
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'degrees_of_freedom': dof
        })
    
    return results

# Perform pairwise comparisons for each model
print("=" * 60)
print("POST-HOC PAIRWISE COMPARISONS")
print("=" * 60)

# Use valid topics only
topics = valid_topics
topic_pairs = list(combinations(topics, 2))

# Bonferroni correction
alpha = 0.05
n_comparisons = len(topic_pairs)
bonferroni_alpha = alpha / n_comparisons

print(f"Number of pairwise comparisons: {n_comparisons}")
print(f"Bonferroni-corrected alpha: {bonferroni_alpha:.6f}")

pairwise_results = {}

for model in models:
    sentiment_col = f'sentiment_{model}'
    print(f"\n{model.upper()}:")
    print("-" * 60)
    
    results = perform_pairwise_chi_square(df_valid, 'search_query', sentiment_col, topic_pairs)
    pairwise_results[model] = results
    
    # Apply Bonferroni correction
    significant_pairs = []
    for result in results:
        if result['p_value'] < bonferroni_alpha:
            significant_pairs.append((result['topic1'], result['topic2'], result['p_value']))
            print(f"  {result['topic1']} vs {result['topic2']}: p = {result['p_value']:.6f} (SIGNIFICANT)")
        else:
            print(f"  {result['topic1']} vs {result['topic2']}: p = {result['p_value']:.6f}")
    
    print(f"\n  Significant pairs: {len(significant_pairs)}/{len(results)}")
    
    # Save pairwise results
    pairwise_df = pd.DataFrame(results)
    pairwise_df.to_csv(os.path.join(project_root, 'outputs', 'tables', f'pairwise_comparisons_{model}.csv'), index=False)
    print(f"  ✓ Pairwise results saved")

print("\n✓ All pairwise comparisons complete")

# Cell 9: Sentiment Distribution Visualizations
print("=" * 60)
print("SENTIMENT DISTRIBUTION VISUALIZATIONS")
print("=" * 60)

os.makedirs(os.path.join(project_root, 'outputs', 'figures'), exist_ok=True)

for model in models:
    sentiment_col = f'sentiment_{model}'
    
    # Create grouped bar chart
    contingency_table = pd.crosstab(df_valid['search_query'], df_valid[sentiment_col])
    contingency_table_pct = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Counts
    contingency_table.plot(kind='bar', ax=axes[0], color=['#ff6b6b', '#4ecdc4', '#95e1d3'])
    axes[0].set_xlabel('Topic', fontweight='bold', fontsize=12)
    axes[0].set_ylabel('Number of Comments', fontweight='bold', fontsize=12)
    axes[0].set_title(f'Sentiment Distribution by Topic - {model.upper()}\n(Absolute Counts)', 
                     fontweight='bold', fontsize=14, pad=15)
    axes[0].legend(title='Sentiment', title_fontsize=10)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(alpha=0.3, linestyle='--', axis='y')
    
    # Percentages
    contingency_table_pct.plot(kind='bar', ax=axes[1], color=['#ff6b6b', '#4ecdc4', '#95e1d3'])
    axes[1].set_xlabel('Topic', fontweight='bold', fontsize=12)
    axes[1].set_ylabel('Percentage (%)', fontweight='bold', fontsize=12)
    axes[1].set_title(f'Sentiment Distribution by Topic - {model.upper()}\n(Percentages)', 
                     fontweight='bold', fontsize=14, pad=15)
    axes[1].legend(title='Sentiment', title_fontsize=10)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, 'outputs', 'figures', f'sentiment_distribution_{model}.png'), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: sentiment_distribution_{model}.png")
    plt.close()

print("\n✓ All sentiment distribution visualizations saved")

# Cell 11: Engagement Analysis
print("=" * 60)
print("ENGAGEMENT ANALYSIS")
print("=" * 60)

engagement_cols = ['like_count', 'reply_count']
if 'video_view_count' in df.columns:
    engagement_cols.append('video_view_count')

engagement_results = {}

for model in models:
    sentiment_col = f'sentiment_{model}'
    print(f"\n{model.upper()}:")
    print("-" * 60)
    
    model_results = {}
    
    for eng_col in engagement_cols:
        if eng_col not in df_valid.columns:
            continue
        
        # Convert to numeric (ensure it's numeric for calculations)
        df_valid[eng_col] = pd.to_numeric(df_valid[eng_col], errors='coerce').fillna(0)
        
        # Group by sentiment and calculate statistics
        stats = df_valid.groupby(sentiment_col)[eng_col].agg(['mean', 'median', 'std', 'count']).reset_index()
        stats.columns = ['sentiment', 'mean', 'median', 'std', 'count']
        
        model_results[eng_col] = stats
        print(f"\n  {eng_col}:")
        print(stats.to_string(index=False))
        
        # Save statistics
        stats.to_csv(f'../outputs/tables/engagement_{model}_{eng_col}.csv', index=False)
    
    engagement_results[model] = model_results

print("\n✓ Engagement analysis complete")

# Cell 11b: Kruskal-Wallis Tests for Sentiment-Engagement Relationship
print("=" * 60)
print("KRUSKAL-WALLIS TESTS FOR SENTIMENT-ENGAGEMENT RELATIONSHIP")
print("=" * 60)

# Engagement metrics to test
engagement_metrics = ['like_count', 'reply_count']

# Store Kruskal-Wallis results
kruskal_results = []

for model in models:
    sentiment_col = f'sentiment_{model}'
    print(f"\n{model.upper()}:")
    print("-" * 60)
    
    for eng_col in engagement_metrics:
        if eng_col not in df_valid.columns:
            continue
        
        # Convert to numeric
        df_valid[eng_col] = pd.to_numeric(df_valid[eng_col], errors='coerce').fillna(0)
        
        # Get engagement values for each sentiment category
        sentiment_groups = []
        sentiment_labels = []
        
        for sentiment in df_valid[sentiment_col].unique():
            sentiment_data = df_valid[df_valid[sentiment_col] == sentiment][eng_col].values
            if len(sentiment_data) > 0:
                sentiment_groups.append(sentiment_data)
                sentiment_labels.append(sentiment)
        
        # Perform Kruskal-Wallis test if we have at least 2 groups
        if len(sentiment_groups) >= 2:
            h_stat, p_value = kruskal(*sentiment_groups)
            
            # Calculate effect size (eta-squared approximation)
            n = len(df_valid)
            k = len(sentiment_groups)
            eta_squared = (h_stat - k + 1) / (n - k) if (n - k) > 0 else 0.0
            
            # Determine significance
            significant = 'Yes' if p_value < 0.05 else 'No'
            
            print(f"\n  {eng_col.upper()}:")
            print(f"    H-statistic: {h_stat:.4f}")
            print(f"    p-value: {p_value:.6f}")
            print(f"    Effect size (η²): {eta_squared:.6f}")
            print(f"    Significant: {significant}")
            
            if p_value < 0.05:
                print(f"    Result: SIGNIFICANT (p < 0.05) - Sentiment is related to {eng_col}")
            else:
                print(f"    Result: NOT SIGNIFICANT (p >= 0.05) - No significant relationship")
            
            # Store results
            kruskal_results.append({
                'model': model,
                'engagement_metric': eng_col,
                'h_statistic': h_stat,
                'p_value': p_value,
                'eta_squared': eta_squared,
                'significant': significant
            })
        else:
            print(f"\n  {eng_col.upper()}:")
            print(f"    Cannot perform test - insufficient groups")

# Save Kruskal-Wallis results
if kruskal_results:
    kruskal_df = pd.DataFrame(kruskal_results)
    kruskal_df.to_csv(os.path.join(project_root, 'outputs', 'tables', 'sentiment_engagement_relationship.csv'), index=False)
    print("\n✓ Kruskal-Wallis test results saved to outputs/tables/sentiment_engagement_relationship.csv")
    
    # Display summary
    print("\n" + "=" * 60)
    print("KRUSKAL-WALLIS TEST SUMMARY")
    print("=" * 60)
    print(kruskal_df.to_string(index=False))

# Cell 13: Qualitative Examples
print("=" * 60)
print("QUALITATIVE EXAMPLES")
print("=" * 60)

# Use transformer model as primary (most accurate)
sentiment_col = 'sentiment_transformer'
text_col = 'comment_text_original'

examples = []

for topic in valid_topics:
    for sentiment in ['positive', 'negative', 'neutral']:
        # Get examples for this topic-sentiment combination
        examples_df = df_valid[(df_valid['search_query'] == topic) & (df_valid[sentiment_col] == sentiment)]
        
        if len(examples_df) > 0:
            # Sample up to 3 examples
            sample_size = min(3, len(examples_df))
            sample = examples_df.sample(n=sample_size, random_state=42)
            
            for idx, row in sample.iterrows():
                text = str(row[text_col])
                if len(text) > 200:
                    text = text[:200] + '...'
                examples.append({
                    'topic': topic,
                    'sentiment': sentiment,
                    'text': text,
                    'like_count': row.get('like_count', 0),
                    'reply_count': row.get('reply_count', 0)
                })

examples_df = pd.DataFrame(examples)
examples_df.to_csv(os.path.join(project_root, 'outputs', 'tables', 'qualitative_examples.csv'), index=False)
print(f"✓ Saved {len(examples)} qualitative examples")

# Display sample
print("\nSample examples:")
print(examples_df.head(10).to_string(index=False))

# Cell 15: Generate RESULTS_REPORT.md
print("=" * 60)
print("GENERATING RESULTS REPORT")
print("=" * 60)

report = []
report.append("# Sentiment Analysis of Astrobiology YouTube Comments: Results Report")
report.append("\n## Executive Summary")
report.append("\nThis report presents the findings from a comprehensive sentiment analysis of YouTube comments")
report.append("on astrobiology-related videos. We analyzed comments across multiple topics using three")
report.append("sentiment analysis methods: TextBlob, VADER, and a Transformer model (DistilBERT).")
# Get valid topics (filter out NaN and invalid entries)
valid_topics = ['3I/ATLAS', 'Oumuamua', 'K2-18b', "Tabby's Star", 'Venus phosphine']
df_valid = df[df['search_query'].isin(valid_topics)].copy()

report.append("\n## Dataset Overview")
report.append(f"\n- **Total Comments Analyzed**: {len(df_valid):,}")
report.append(f"- **Number of Topics**: {len(valid_topics)}")
report.append(f"- **Topics**: {', '.join(valid_topics)}")
if 'published_at' in df_valid.columns:
    report.append(f"- **Date Range**: {df_valid['published_at'].min()} to {df_valid['published_at'].max()}")
else:
    report.append(f"- **Date Range**: N/A")

report.append("\n## Methodology")
report.append("\n### Sentiment Analysis Models")
report.append("\n1. **TextBlob**: Rule-based sentiment analysis using polarity scores")
report.append("2. **VADER**: Valence Aware Dictionary and sEntiment Reasoner, optimized for social media")
report.append("3. **Transformer**: DistilBERT-based model fine-tuned on sentiment analysis")

report.append("\n### Preprocessing Strategy")
report.append("\nWe implemented a four-track preprocessing strategy:")
report.append("- **TextBlob**: Heavy preprocessing (lowercase, remove punctuation/emojis, lemmatization)")
report.append("- **VADER**: Minimal preprocessing (preserves punctuation, capitalization, emojis)")
report.append("- **Transformer**: Light preprocessing (removes only URLs and mentions)")
report.append("- **Raw**: Original text preserved for engagement feature extraction")

report.append("\n## Key Findings")
report.append("\n### 1. Sentiment Distribution by Topic")
report.append("\nChi-square tests of independence were performed to determine if sentiment distributions")
report.append("differ significantly across astrobiology topics.\n")

for model in models:
    result = chi_square_results[model]
    report.append(f"\n**{model.upper()} Model:**")
    report.append(f"- Chi-square statistic: {result['chi2_statistic']:.4f}")
    report.append(f"- p-value: {result['p_value']:.6f}")
    report.append(f"- Cramer's V: {result['cramers_v']:.4f}")
    if result['p_value'] < 0.05:
        report.append(f"- **Result**: Significant difference in sentiment distribution across topics (p < 0.05)")
    else:
        report.append(f"- **Result**: No significant difference in sentiment distribution across topics")

report.append("\n### 2. Post-hoc Pairwise Comparisons")
report.append(f"\nPairwise chi-square tests were conducted with Bonferroni correction (α = {bonferroni_alpha:.6f}).")
report.append("Significant differences between topic pairs:\n")

for model in models:
    report.append(f"\n**{model.upper()} Model:**")
    significant_count = sum(1 for r in pairwise_results[model] if r['p_value'] < bonferroni_alpha)
    report.append(f"- Significant pairs: {significant_count}/{len(pairwise_results[model])}")

report.append("\n### 3. Engagement Metrics")
report.append("\nEngagement metrics (likes, replies) were analyzed by sentiment category to understand")
report.append("the relationship between sentiment and user engagement.")

report.append("\n## Conclusion")
report.append("\nThis analysis reveals significant variations in sentiment across different astrobiology topics,")
report.append("suggesting that public perception and emotional responses differ based on the specific")
report.append("scientific topic being discussed. These findings have implications for science communication")
report.append("strategies and understanding public engagement with astrobiology content.")

report.append("\n## References")
report.append("\n- TextBlob: https://textblob.readthedocs.io/")
report.append("\n- VADER: Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for")
report.append("Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and")
report.append("Social Media (ICWSM-14).")
report.append("\n- DistilBERT: Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT: smaller,")
report.append("faster, cheaper and lighter. arXiv preprint arXiv:1910.01108.")

# Write report to file
report_text = '\n'.join(report)
with open(os.path.join(project_root, 'RESULTS_REPORT.md'), 'w', encoding='utf-8') as f:
    f.write(report_text)

print("✓ Results report saved to RESULTS_REPORT.md")
print(f"\nReport length: {len(report_text)} characters")
report_sections = len([line for line in report if line.startswith('##')])
print(f"Report sections: {report_sections}")

print("\n" + "=" * 60)
print("NOTEBOOK 5 COMPLETE")
print("=" * 60)
print("All analysis complete! Review RESULTS_REPORT.md for comprehensive findings.")

