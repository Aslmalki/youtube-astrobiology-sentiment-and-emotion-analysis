#!/usr/bin/env python3
"""
Emotion Analysis: Outliers vs Typical Comments
This script analyzes which emotions dominate among outlier comments compared to typical comments.

Addresses reviewer question: "Would it be interesting to look at emotions that dominate 
among the outlier comments?"
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Get project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

print("=" * 80)
print("EMOTION ANALYSIS: OUTLIERS vs TYPICAL COMMENTS")
print("=" * 80)

# Load emotion data
print("\nLoading emotion data...")
emotion_data_path = os.path.join(project_root, 'data/processed/youtube_comments_with_emotions.csv')
if not os.path.exists(emotion_data_path):
    raise FileNotFoundError(f"Emotion data not found at: {emotion_data_path}")

df_emotions = pd.read_csv(emotion_data_path)
print(f"Loaded {len(df_emotions):,} comments with emotion data")

# Check required columns
required_cols = ['search_query', 'dominant_emotion', 'like_count', 'reply_count']
for col in required_cols:
    if col not in df_emotions.columns:
        raise ValueError(f"'{col}' column not found in dataset")

# Ensure numeric columns are numeric
df_emotions['like_count'] = pd.to_numeric(df_emotions['like_count'], errors='coerce').fillna(0)
df_emotions['reply_count'] = pd.to_numeric(df_emotions['reply_count'], errors='coerce').fillna(0)

# Identify outliers (same method as sentiment analysis)
print("\nIdentifying outliers...")
like_threshold = df_emotions['like_count'].quantile(0.99)
reply_threshold = df_emotions['reply_count'].quantile(0.99)

print(f"99th Percentile Thresholds:")
print(f"  Likes: {like_threshold:.0f}")
print(f"  Replies: {reply_threshold:.0f}")

# Identify outliers: comments with likes OR replies > 99th percentile
df_emotions['comment_type'] = 'typical'
df_emotions.loc[(df_emotions['like_count'] > like_threshold) | 
                (df_emotions['reply_count'] > reply_threshold), 'comment_type'] = 'outlier'

outliers = df_emotions[df_emotions['comment_type'] == 'outlier'].copy()
typical = df_emotions[df_emotions['comment_type'] == 'typical'].copy()

print(f"\nComment Type Distribution:")
print(f"  Outliers: {len(outliers):,} ({len(outliers)/len(df_emotions)*100:.2f}%)")
print(f"  Typical: {len(typical):,} ({len(typical)/len(df_emotions)*100:.2f}%)")

# ============================================================================
# EMOTION DISTRIBUTION COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("EMOTION DISTRIBUTION: OUTLIERS vs TYPICAL")
print("=" * 80)

# Calculate emotion distribution for outliers and typical comments
outlier_emotions = outliers['dominant_emotion'].value_counts()
typical_emotions = typical['dominant_emotion'].value_counts()

# Exclude neutral if present
if 'neutral' in outlier_emotions.index:
    outlier_emotions = outlier_emotions.drop('neutral')
if 'neutral' in typical_emotions.index:
    typical_emotions = typical_emotions.drop('neutral')

# Get all emotions (union of both)
all_emotions = sorted(set(outlier_emotions.index.tolist() + typical_emotions.index.tolist()))

# Calculate percentages
outlier_emotions_pct = (outlier_emotions / len(outliers) * 100).round(2)
typical_emotions_pct = (typical_emotions / len(typical) * 100).round(2)

# Create comparison dataframe
comparison_data = []
for emotion in all_emotions:
    outlier_count = outlier_emotions.get(emotion, 0)
    typical_count = typical_emotions.get(emotion, 0)
    outlier_pct = outlier_emotions_pct.get(emotion, 0)
    typical_pct = typical_emotions_pct.get(emotion, 0)
    difference = outlier_pct - typical_pct
    
    comparison_data.append({
        'emotion': emotion,
        'outlier_count': outlier_count,
        'outlier_pct': outlier_pct,
        'typical_count': typical_count,
        'typical_pct': typical_pct,
        'difference_pct': difference,
        'outlier_ratio': outlier_pct / typical_pct if typical_pct > 0 else np.nan
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('outlier_pct', ascending=False)

print("\nTop 15 Emotions in Outliers (by percentage):")
print(comparison_df.head(15)[['emotion', 'outlier_pct', 'typical_pct', 'difference_pct']].to_string(index=False))

# Find emotions that are overrepresented in outliers
overrepresented = comparison_df[comparison_df['difference_pct'] > 1.0].copy()
overrepresented = overrepresented.sort_values('difference_pct', ascending=False)

print(f"\n\nEmotions Overrepresented in Outliers (difference > 1 percentage point):")
print(f"Found {len(overrepresented)} emotions")
if len(overrepresented) > 0:
    print(overrepresented[['emotion', 'outlier_pct', 'typical_pct', 'difference_pct']].to_string(index=False))

# Find emotions that are underrepresented in outliers
underrepresented = comparison_df[comparison_df['difference_pct'] < -1.0].copy()
underrepresented = underrepresented.sort_values('difference_pct', ascending=True)

print(f"\n\nEmotions Underrepresented in Outliers (difference < -1 percentage point):")
print(f"Found {len(underrepresented)} emotions")
if len(underrepresented) > 0:
    print(underrepresented[['emotion', 'outlier_pct', 'typical_pct', 'difference_pct']].to_string(index=False))

# ============================================================================
# CHI-SQUARE TEST: OUTLIER vs TYPICAL × EMOTION
# ============================================================================
print("\n" + "=" * 80)
print("CHI-SQUARE TEST: OUTLIER vs TYPICAL × EMOTION")
print("=" * 80)

# Filter to top emotions (to avoid sparse cells)
top_emotions = comparison_df.head(15)['emotion'].tolist()
df_top_emotions = df_emotions[df_emotions['dominant_emotion'].isin(top_emotions)].copy()

# Create contingency table
contingency_table = pd.crosstab(df_top_emotions['comment_type'], 
                                df_top_emotions['dominant_emotion'])

print("\nContingency Table (Outlier vs Typical × Top 15 Emotions):")
print(contingency_table)

# Perform chi-square test
chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

print(f"\nChi-square statistic: {chi2_stat:.4f}")
print(f"Degrees of freedom: {dof}")
print(f"P-value: {p_value:.6f}")

# Calculate effect size (Cramér's V)
n = contingency_table.sum().sum()
cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))

print(f"Effect Size (Cramér's V): {cramers_v:.4f}")
if cramers_v < 0.1:
    effect_interpretation = "negligible"
elif cramers_v < 0.3:
    effect_interpretation = "small"
elif cramers_v < 0.5:
    effect_interpretation = "medium"
else:
    effect_interpretation = "large"
print(f"Effect Size Interpretation: {effect_interpretation}")

if p_value < 0.05:
    print(f"\n✓ Statistically significant association (p < 0.05)")
    print(f"  The emotion distribution differs significantly between outliers and typical comments.")
else:
    print(f"\n✗ No statistically significant association (p >= 0.05)")
    print(f"  The emotion distribution does not differ significantly between outliers and typical comments.")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

output_dir = os.path.join(project_root, 'outputs/figures')
os.makedirs(output_dir, exist_ok=True)

# Figure 1: Comparison of top emotions
top_15_for_viz = comparison_df.head(15)
fig, ax = plt.subplots(figsize=(12, 8))

x = np.arange(len(top_15_for_viz))
width = 0.35

bars1 = ax.barh(x - width/2, top_15_for_viz['outlier_pct'], width, 
                label='Outliers', color='#e74c3c', alpha=0.8)
bars2 = ax.barh(x + width/2, top_15_for_viz['typical_pct'], width, 
                label='Typical', color='#3498db', alpha=0.8)

ax.set_xlabel('Percentage (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Emotion', fontsize=12, fontweight='bold')
ax.set_title('Top 15 Emotions: Outliers vs Typical Comments', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_yticks(x)
ax.set_yticklabels(top_15_for_viz['emotion'])
ax.legend()
ax.grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'emotion_outlier_comparison.png'), 
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: outputs/figures/emotion_outlier_comparison.png")

# Figure 2: Difference plot (outliers - typical)
top_20_diff = comparison_df.nlargest(20, 'outlier_pct')
fig, ax = plt.subplots(figsize=(10, 8))

colors = ['#e74c3c' if x > 0 else '#3498db' for x in top_20_diff['difference_pct']]
bars = ax.barh(range(len(top_20_diff)), top_20_diff['difference_pct'], color=colors, alpha=0.8)

ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
ax.set_xlabel('Difference in Percentage Points (Outliers - Typical)', 
              fontsize=12, fontweight='bold')
ax.set_ylabel('Emotion', fontsize=12, fontweight='bold')
ax.set_title('Emotion Differences: Outliers vs Typical Comments', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_yticks(range(len(top_20_diff)))
ax.set_yticklabels(top_20_diff['emotion'])
ax.grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'emotion_outlier_difference.png'), 
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: outputs/figures/emotion_outlier_difference.png")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

tables_dir = os.path.join(project_root, 'outputs/tables')
os.makedirs(tables_dir, exist_ok=True)

# Save comparison table
comparison_df.to_csv(os.path.join(tables_dir, 'emotion_outlier_comparison.csv'), index=False)
print(f"✓ Saved: outputs/tables/emotion_outlier_comparison.csv")

# Save contingency table
contingency_table.to_csv(os.path.join(tables_dir, 'emotion_outlier_contingency.csv'))
print(f"✓ Saved: outputs/tables/emotion_outlier_contingency.csv")

# Save summary statistics
summary_results = {
    'chi2_statistic': chi2_stat,
    'p_value': p_value,
    'degrees_of_freedom': dof,
    'cramers_v': cramers_v,
    'effect_size_interpretation': effect_interpretation,
    'n_outliers': len(outliers),
    'n_typical': len(typical),
    'n_top_emotions_tested': len(top_emotions)
}

summary_df = pd.DataFrame([summary_results])
summary_df.to_csv(os.path.join(tables_dir, 'emotion_outlier_chi_square_results.csv'), index=False)
print(f"✓ Saved: outputs/tables/emotion_outlier_chi_square_results.csv")

# ============================================================================
# GENERATE REPORT
# ============================================================================
print("\n" + "=" * 80)
print("GENERATING REPORT")
print("=" * 80)

report_lines = []
report_lines.append("=" * 80)
report_lines.append("EMOTION ANALYSIS: OUTLIERS vs TYPICAL COMMENTS")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append("Date: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
report_lines.append("")
report_lines.append("=" * 80)
report_lines.append("SUMMARY")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append(f"Outliers: {len(outliers):,} comments ({len(outliers)/len(df_emotions)*100:.2f}%)")
report_lines.append(f"Typical: {len(typical):,} comments ({len(typical)/len(df_emotions)*100:.2f}%)")
report_lines.append("")
report_lines.append("=" * 80)
report_lines.append("TOP EMOTIONS IN OUTLIERS")
report_lines.append("=" * 80)
report_lines.append("")
for i, row in comparison_df.head(10).iterrows():
    report_lines.append(f"{i+1}. {row['emotion']}: {row['outlier_pct']:.2f}% (vs. {row['typical_pct']:.2f}% in typical, diff: {row['difference_pct']:+.2f}%)")
report_lines.append("")
report_lines.append("=" * 80)
report_lines.append("OVERREPRESENTED EMOTIONS IN OUTLIERS")
report_lines.append("=" * 80)
report_lines.append("")
if len(overrepresented) > 0:
    for i, row in overrepresented.head(10).iterrows():
        report_lines.append(f"• {row['emotion']}: {row['outlier_pct']:.2f}% in outliers vs. {row['typical_pct']:.2f}% in typical (diff: {row['difference_pct']:+.2f}%)")
else:
    report_lines.append("No emotions are overrepresented by more than 1 percentage point.")
report_lines.append("")
report_lines.append("=" * 80)
report_lines.append("UNDERREPRESENTED EMOTIONS IN OUTLIERS")
report_lines.append("=" * 80)
report_lines.append("")
if len(underrepresented) > 0:
    for i, row in underrepresented.head(10).iterrows():
        report_lines.append(f"• {row['emotion']}: {row['outlier_pct']:.2f}% in outliers vs. {row['typical_pct']:.2f}% in typical (diff: {row['difference_pct']:+.2f}%)")
else:
    report_lines.append("No emotions are underrepresented by more than 1 percentage point.")
report_lines.append("")
report_lines.append("=" * 80)
report_lines.append("CHI-SQUARE TEST RESULTS")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append(f"Chi-square statistic: {chi2_stat:.4f}")
report_lines.append(f"Degrees of freedom: {dof}")
report_lines.append(f"P-value: {p_value:.6f}")
report_lines.append(f"Effect size (Cramér's V): {cramers_v:.4f} ({effect_interpretation})")
report_lines.append("")
if p_value < 0.05:
    report_lines.append("CONCLUSION: The emotion distribution differs significantly between")
    report_lines.append("outliers and typical comments.")
else:
    report_lines.append("CONCLUSION: The emotion distribution does not differ significantly")
    report_lines.append("between outliers and typical comments.")
report_lines.append("")
report_lines.append("=" * 80)
report_lines.append("RECOMMENDATIONS FOR MANUSCRIPT")
report_lines.append("=" * 80)
report_lines.append("")
if len(overrepresented) > 0:
    top_over = overrepresented.iloc[0]
    report_lines.append("1. Consider adding a brief section or paragraph discussing which")
    report_lines.append("   emotions are associated with high engagement (outliers).")
    report_lines.append("")
    report_lines.append(f"   Example text:")
    report_lines.append(f"   'Analysis of emotions in outlier comments (n={len(outliers):,}) revealed")
    report_lines.append(f"   that {top_over['emotion']} was the most overrepresented emotion")
    report_lines.append(f"   ({top_over['outlier_pct']:.2f}% in outliers vs. {top_over['typical_pct']:.2f}% in typical comments).'")
    report_lines.append("")
    report_lines.append("2. This analysis complements the sentiment analysis (outliers vs typical)")
    report_lines.append("   by providing more granular insight into which specific emotions")
    report_lines.append("   drive high engagement.")
else:
    report_lines.append("1. The emotion distribution is similar between outliers and typical comments.")
    report_lines.append("   This suggests that emotions alone may not be strong predictors of")
    report_lines.append("   outlier status, consistent with the weak emotion-engagement relationships")
    report_lines.append("   found in the regression analysis.")
report_lines.append("")
report_lines.append("=" * 80)

# Save report
report_path = os.path.join(project_root, 'outputs/emotion_outlier_analysis_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print(f"\n✓ Report saved to: {report_path}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nKey Findings:")
print(f"  - Top emotion in outliers: {comparison_df.iloc[0]['emotion']} ({comparison_df.iloc[0]['outlier_pct']:.2f}%)")
print(f"  - Emotions overrepresented in outliers: {len(overrepresented)}")
print(f"  - Chi-square test: χ²({dof}) = {chi2_stat:.4f}, p = {p_value:.6f}")
print(f"  - Effect size: Cramér's V = {cramers_v:.4f} ({effect_interpretation})")
