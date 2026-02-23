#!/usr/bin/env python3
"""
Test Normality Per Topic
========================
Tests whether normality is rejected for engagement metrics separately for each topic.
This helps answer: "Does the normality rejection apply equally to all 5 topics?"
"""

import pandas as pd
import numpy as np
from scipy.stats import shapiro, normaltest, jarque_bera
import os

print("="*80)
print("NORMALITY TESTS PER TOPIC")
print("="*80)

# Load data
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_path = os.path.join(project_root, 'data/processed/03_sentiment_results.csv')

df = pd.read_csv(data_path)
print(f"\nLoaded {len(df):,} comments")

# Filter to valid topics
valid_topics = ['3I/ATLAS', 'Oumuamua', 'K2-18b', "Tabby's Star", 'Venus phosphine']
df = df[df['search_query'].isin(valid_topics)].copy()
print(f"Filtered to {len(df):,} comments with valid topics")

# Ensure numeric
for col in ['like_count', 'reply_count']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int32')

# Test normality per topic
engagement_metrics = ['like_count', 'reply_count']
results = []

for topic in valid_topics:
    topic_df = df[df['search_query'] == topic].copy()
    print(f"\n{'='*80}")
    print(f"TOPIC: {topic} (n = {len(topic_df):,})")
    print(f"{'='*80}")
    
    for metric in engagement_metrics:
        data = topic_df[metric].values
        
        # Sample if too large (Shapiro-Wilk limit is ~5000)
        if len(data) > 5000:
            data_sample = np.random.choice(data, size=5000, replace=False)
        else:
            data_sample = data
        
        # Calculate basic stats
        zero_pct = (data == 0).sum() / len(data) * 100
        skewness = pd.Series(data).skew()
        kurtosis = pd.Series(data).kurtosis()
        
        # Normality tests
        try:
            shapiro_stat, shapiro_p = shapiro(data_sample) if len(data_sample) <= 5000 else (np.nan, np.nan)
        except:
            shapiro_stat, shapiro_p = np.nan, np.nan
        
        dagostino_stat, dagostino_p = normaltest(data_sample)
        jb_stat, jb_p = jarque_bera(data_sample)
        
        # Determine if normal (p > 0.05 means fail to reject normality)
        is_normal = False
        if not np.isnan(shapiro_p) and shapiro_p > 0.05:
            is_normal = True
        
        # Store results
        result = {
            'topic': topic,
            'metric': metric,
            'n': len(data),
            'zero_pct': zero_pct,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'shapiro_p': shapiro_p,
            'dagostino_p': dagostino_p,
            'jarque_bera_p': jb_p,
            'rejects_normality': (shapiro_p < 0.001 if not np.isnan(shapiro_p) else False) and 
                                 (dagostino_p < 0.001) and (jb_p < 0.001)
        }
        results.append(result)
        
        # Print results
        print(f"\n  {metric.upper()}:")
        print(f"    Zero percentage: {zero_pct:.1f}%")
        print(f"    Skewness: {skewness:.3f}")
        print(f"    Shapiro-Wilk: p = {shapiro_p:.6f} {'(rejects normality)' if shapiro_p < 0.001 else '(fails to reject)' if not np.isnan(shapiro_p) else '(N/A)'}")
        print(f"    D'Agostino: p = {dagostino_p:.6f} {'(rejects normality)' if dagostino_p < 0.001 else '(fails to reject)'}")
        print(f"    Jarque-Bera: p = {jb_p:.6f} {'(rejects normality)' if jb_p < 0.001 else '(fails to reject)'}")
        print(f"    All tests reject normality (p < 0.001): {result['rejects_normality']}")

# Create summary DataFrame
results_df = pd.DataFrame(results)

# Save results
output_path = os.path.join(project_root, 'outputs/tables/normality_tests_per_topic.csv')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
results_df.to_csv(output_path, index=False)
print(f"\n{'='*80}")
print(f"✓ Results saved to: {output_path}")

# Summary: Check if all topics reject normality
print(f"\n{'='*80}")
print("SUMMARY: Does normality rejection apply equally to all topics?")
print(f"{'='*80}")

for metric in engagement_metrics:
    metric_results = results_df[results_df['metric'] == metric]
    all_reject = metric_results['rejects_normality'].all()
    
    print(f"\n{metric.upper()}:")
    print(f"  All 5 topics reject normality (p < 0.001): {all_reject}")
    
    if not all_reject:
        print(f"  Topics that DO NOT reject normality:")
        for _, row in metric_results[~metric_results['rejects_normality']].iterrows():
            print(f"    - {row['topic']}: Shapiro={row['shapiro_p']:.6f}, D'Agostino={row['dagostino_p']:.6f}, JB={row['jarque_bera_p']:.6f}")
    else:
        print(f"  ✓ All topics strongly reject normality (p < 0.001)")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")
