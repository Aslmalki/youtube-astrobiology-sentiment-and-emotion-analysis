#!/usr/bin/env python3
"""
RQ2: Statistical Tests for Engagement Patterns Across Topics
============================================================
Performs Kruskal-Wallis tests to determine if engagement patterns
(comment length, word count, question marks, exclamation marks) 
differ significantly across the five astrobiology topics.
"""

import pandas as pd
import numpy as np
from scipy.stats import kruskal
import os

print("="*80)
print("RQ2: STATISTICAL TESTS FOR ENGAGEMENT PATTERNS ACROSS TOPICS")
print("="*80)

# Get project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Load data
data_path = os.path.join(project_root, 'data/processed/02_preprocessed_data.csv')
print(f"\nLoading data from: {data_path}")
df = pd.read_csv(data_path)
print(f"✓ Loaded {len(df):,} comments")

# Filter to valid topics
valid_topics = ['3I/ATLAS', 'Oumuamua', 'K2-18b', "Tabby's Star", 'Venus phosphine']
df = df[df['search_query'].isin(valid_topics)].copy()
print(f"✓ Filtered to {len(df):,} comments with valid topics")

# Ensure numeric columns
numeric_cols = ['text_length', 'word_count', 'question_count', 'exclamation_count']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Prepare data for Kruskal-Wallis tests
print("\n" + "="*80)
print("PERFORMING KRUSKAL-WALLIS TESTS")
print("="*80)

results = []

# Metrics to test
metrics = {
    'text_length': 'Comment Length (characters)',
    'word_count': 'Word Count',
    'question_count': 'Question Marks per Comment',
    'exclamation_count': 'Exclamation Marks per Comment'
}

for col, metric_name in metrics.items():
    if col not in df.columns:
        print(f"⚠ Warning: {col} not found in data")
        continue
    
    print(f"\n{metric_name.upper()}:")
    print("-" * 80)
    
    # Prepare groups for each topic
    groups = []
    group_names = []
    
    for topic in valid_topics:
        topic_data = df[df['search_query'] == topic][col].dropna()
        if len(topic_data) > 0:
            groups.append(topic_data.values)
            group_names.append(topic)
            print(f"  {topic}: n={len(topic_data):,}, median={topic_data.median():.2f}, mean={topic_data.mean():.2f}")
    
    if len(groups) < 2:
        print(f"  ⚠ Skipping: Need at least 2 groups")
        continue
    
    # Perform Kruskal-Wallis test
    try:
        h_stat, p_value = kruskal(*groups)
        
        # Calculate effect size (eta-squared approximation)
        # η² = (H - k + 1) / (N - k)
        # where H is the H-statistic, k is number of groups, N is total sample size
        total_n = sum(len(g) for g in groups)
        k = len(groups)
        eta_squared = (h_stat - k + 1) / (total_n - k) if (total_n - k) > 0 else 0
        
        # Determine significance
        significant = p_value < 0.05
        
        # Store results
        result = {
            'metric': metric_name,
            'h_statistic': h_stat,
            'p_value': p_value,
            'eta_squared': eta_squared,
            'significant': significant,
            'n_total': total_n,
            'n_groups': k
        }
        results.append(result)
        
        # Print results
        print(f"  Kruskal-Wallis H-statistic: {h_stat:.4f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Effect size (η²): {eta_squared:.6f}")
        print(f"  Significant: {'Yes' if significant else 'No'} (p {'<' if p_value < 0.05 else '>='} 0.05)")
        
        # If significant, note that post-hoc tests would be needed
        if significant:
            print(f"  → Post-hoc pairwise comparisons (e.g., Dunn's test) would identify")
            print(f"    which specific topic pairs differ significantly.")
        
    except Exception as e:
        print(f"  ❌ Error performing test: {str(e)}")
        continue

# Save results
if results:
    results_df = pd.DataFrame(results)
    output_path = os.path.join(project_root, 'outputs/tables/rq2_engagement_patterns_kruskal_wallis.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for result in results:
        sig_marker = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['significant'] else ""
        print(f"{result['metric']}: H={result['h_statistic']:.2f}, p={result['p_value']:.6f}, η²={result['eta_squared']:.6f} {sig_marker}")
else:
    print("\n⚠ No results to save")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
