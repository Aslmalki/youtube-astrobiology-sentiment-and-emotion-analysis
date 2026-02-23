"""
Chi-Square Test for Independence: Comment Type (Outlier vs Typical) and Sentiment
This script performs a statistical test to determine if there is a significant
association between comment type and sentiment distribution.
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import os
import warnings

warnings.filterwarnings('ignore')

# Get project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

print("=" * 80)
print("CHI-SQUARE TEST: COMMENT TYPE vs SENTIMENT")
print("=" * 80)

# Load data
print("\nLoading data...")
data_path = os.path.join(project_root, 'data/processed/03_sentiment_results.csv')
df = pd.read_csv(data_path)
print(f"Loaded {len(df):,} comments")

# Ensure numeric columns are numeric
numeric_cols = ['like_count', 'reply_count']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Identify outliers using the same method as outlier_analysis.py
print("\nIdentifying outliers...")
like_threshold = df['like_count'].quantile(0.99)
reply_threshold = df['reply_count'].quantile(0.99)

print(f"99th Percentile Thresholds:")
print(f"  Likes: {like_threshold:.0f}")
print(f"  Replies: {reply_threshold:.0f}")

# Identify outliers: comments with likes OR replies > 99th percentile
df['comment_type'] = 'typical'
df.loc[(df['like_count'] > like_threshold) | (df['reply_count'] > reply_threshold), 'comment_type'] = 'outlier'

outliers = df[df['comment_type'] == 'outlier'].copy()
typical = df[df['comment_type'] == 'typical'].copy()

print(f"\nComment Type Distribution:")
print(f"  Outliers: {len(outliers):,} ({len(outliers)/len(df)*100:.2f}%)")
print(f"  Typical: {len(typical):,} ({len(typical)/len(df)*100:.2f}%)")

# Check sentiment column
if 'sentiment_transformer' not in df.columns:
    raise ValueError("sentiment_transformer column not found in dataset")

# Check sentiment distribution
print(f"\nSentiment Distribution (All Comments):")
sentiment_counts = df['sentiment_transformer'].value_counts()
for sentiment, count in sentiment_counts.items():
    print(f"  {sentiment}: {count:,} ({count/len(df)*100:.2f}%)")

# Create contingency table for positive vs negative (excluding neutral)
print("\n" + "=" * 80)
print("CHI-SQUARE TEST: OUTLIER vs TYPICAL × POSITIVE vs NEGATIVE")
print("=" * 80)

# Filter to only positive and negative sentiments
df_binary = df[df['sentiment_transformer'].isin(['positive', 'negative'])].copy()
print(f"\nComments with positive/negative sentiment: {len(df_binary):,}")
print(f"  (Excluded {len(df) - len(df_binary):,} neutral comments)")

# Create 2x2 contingency table
contingency_table = pd.crosstab(
    df_binary['comment_type'],
    df_binary['sentiment_transformer'],
    margins=True
)

print("\nContingency Table (Outlier vs Typical × Positive vs Negative):")
print(contingency_table)

# Extract the 2x2 table without margins for chi-square test
contingency_2x2 = pd.crosstab(
    df_binary['comment_type'],
    df_binary['sentiment_transformer']
)

print("\n2x2 Contingency Table (for chi-square test):")
print(contingency_2x2)

# Perform chi-square test
chi2, p_value, dof, expected = chi2_contingency(contingency_2x2)

print("\n" + "=" * 80)
print("CHI-SQUARE TEST RESULTS")
print("=" * 80)
print(f"\nChi-square statistic: {chi2:.4f}")
print(f"Degrees of freedom: {dof}")
print(f"P-value: {p_value:.6f}")
print(f"Alpha level: 0.05")

# Calculate percentages for interpretation
outlier_positive = contingency_2x2.loc['outlier', 'positive']
outlier_negative = contingency_2x2.loc['outlier', 'negative']
outlier_total = outlier_positive + outlier_negative

typical_positive = contingency_2x2.loc['typical', 'positive']
typical_negative = contingency_2x2.loc['typical', 'negative']
typical_total = typical_positive + typical_negative

outlier_positive_pct = (outlier_positive / outlier_total) * 100
outlier_negative_pct = (outlier_negative / outlier_total) * 100
typical_positive_pct = (typical_positive / typical_total) * 100
typical_negative_pct = (typical_negative / typical_total) * 100

print("\nObserved Percentages:")
print(f"  Outliers - Positive: {outlier_positive_pct:.2f}% ({outlier_positive:,}/{outlier_total:,})")
print(f"  Outliers - Negative: {outlier_negative_pct:.2f}% ({outlier_negative:,}/{outlier_total:,})")
print(f"  Typical - Positive: {typical_positive_pct:.2f}% ({typical_positive:,}/{typical_total:,})")
print(f"  Typical - Negative: {typical_negative_pct:.2f}% ({typical_negative:,}/{typical_total:,})")

print("\nExpected Frequencies (under null hypothesis of independence):")
expected_df = pd.DataFrame(
    expected,
    index=contingency_2x2.index,
    columns=contingency_2x2.columns
)
print(expected_df)

# Calculate effect size (Cramér's V)
n = contingency_2x2.sum().sum()
cramers_v = np.sqrt(chi2 / (n * (min(contingency_2x2.shape) - 1)))

print(f"\nEffect Size (Cramér's V): {cramers_v:.4f}")
if cramers_v < 0.1:
    effect_interpretation = "negligible"
elif cramers_v < 0.3:
    effect_interpretation = "small"
elif cramers_v < 0.5:
    effect_interpretation = "medium"
else:
    effect_interpretation = "large"
print(f"Effect Size Interpretation: {effect_interpretation}")

# Statistical conclusion
print("\n" + "=" * 80)
print("STATISTICAL CONCLUSION")
print("=" * 80)
if p_value < 0.05:
    print(f"\nThe chi-square test indicates a statistically significant association")
    print(f"between comment type (outlier vs typical) and sentiment (positive vs negative)")
    print(f"(χ²({dof}) = {chi2:.4f}, p = {p_value:.6f}).")
    print(f"\nThe observed difference in sentiment distribution between outliers and")
    print(f"typical comments is statistically significant at the α = 0.05 level.")
else:
    print(f"\nThe chi-square test does not indicate a statistically significant association")
    print(f"between comment type (outlier vs typical) and sentiment (positive vs negative)")
    print(f"(χ²({dof}) = {chi2:.4f}, p = {p_value:.6f}).")
    print(f"\nThe observed difference in sentiment distribution between outliers and")
    print(f"typical comments is not statistically significant at the α = 0.05 level.")

# Additional analysis: Include neutral sentiment (3x2 table)
print("\n" + "=" * 80)
print("ADDITIONAL ANALYSIS: INCLUDING NEUTRAL SENTIMENT")
print("=" * 80)

# Create 2x3 contingency table (outlier/typical × positive/negative/neutral)
contingency_2x3 = pd.crosstab(
    df['comment_type'],
    df['sentiment_transformer']
)

print("\n2x3 Contingency Table (Outlier vs Typical × Positive vs Negative vs Neutral):")
print(contingency_2x3)

# Perform chi-square test with neutral included
chi2_3cat, p_value_3cat, dof_3cat, expected_3cat = chi2_contingency(contingency_2x3)

print(f"\nChi-square statistic (with neutral): {chi2_3cat:.4f}")
print(f"Degrees of freedom: {dof_3cat}")
print(f"P-value: {p_value_3cat:.6f}")

# Calculate Cramér's V for 3-category test
n_3cat = contingency_2x3.sum().sum()
cramers_v_3cat = np.sqrt(chi2_3cat / (n_3cat * (min(contingency_2x3.shape) - 1)))
print(f"Effect Size (Cramér's V, with neutral): {cramers_v_3cat:.4f}")

if p_value_3cat < 0.05:
    print(f"\nWith neutral sentiment included, the association is statistically significant")
    print(f"(χ²({dof_3cat}) = {chi2_3cat:.4f}, p = {p_value_3cat:.6f}).")
else:
    print(f"\nWith neutral sentiment included, the association is not statistically significant")
    print(f"(χ²({dof_3cat}) = {chi2_3cat:.4f}, p = {p_value_3cat:.6f}).")

# Generate comprehensive report
print("\n" + "=" * 80)
print("GENERATING REPORT")
print("=" * 80)

report_lines = []
report_lines.append("=" * 80)
report_lines.append("CHI-SQUARE TEST REPORT: COMMENT TYPE vs SENTIMENT")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append("Date: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
report_lines.append("")
report_lines.append("=" * 80)
report_lines.append("METHODOLOGY")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append("A chi-square test for independence was performed to determine if there is a")
report_lines.append("statistically significant association between comment type (outlier vs typical)")
report_lines.append("and sentiment (positive vs negative).")
report_lines.append("")
report_lines.append("Outliers were defined as comments with likes or replies exceeding the 99th")
report_lines.append(f"percentile threshold (likes > {like_threshold:.0f} or replies > {reply_threshold:.0f}).")
report_lines.append("")
report_lines.append(f"Dataset: {len(df):,} total comments")
report_lines.append(f"  - Outliers: {len(outliers):,} ({len(outliers)/len(df)*100:.2f}%)")
report_lines.append(f"  - Typical: {len(typical):,} ({len(typical)/len(df)*100:.2f}%)")
report_lines.append("")
report_lines.append("=" * 80)
report_lines.append("RESULTS: POSITIVE vs NEGATIVE SENTIMENT")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append("Contingency Table (2x2):")
report_lines.append("")
for line in str(contingency_2x2).split('\n'):
    report_lines.append("  " + line)
report_lines.append("")
report_lines.append("Observed Percentages:")
report_lines.append(f"  Outliers - Positive: {outlier_positive_pct:.2f}% ({outlier_positive:,}/{outlier_total:,})")
report_lines.append(f"  Outliers - Negative: {outlier_negative_pct:.2f}% ({outlier_negative:,}/{outlier_total:,})")
report_lines.append(f"  Typical - Positive: {typical_positive_pct:.2f}% ({typical_positive:,}/{typical_total:,})")
report_lines.append(f"  Typical - Negative: {typical_negative_pct:.2f}% ({typical_negative:,}/{typical_total:,})")
report_lines.append("")
report_lines.append("Chi-Square Test Statistics:")
report_lines.append(f"  Chi-square statistic (χ²): {chi2:.4f}")
report_lines.append(f"  Degrees of freedom: {dof}")
report_lines.append(f"  P-value: {p_value:.6f}")
report_lines.append(f"  Alpha level: 0.05")
report_lines.append(f"  Effect size (Cramér's V): {cramers_v:.4f} ({effect_interpretation})")
report_lines.append("")
report_lines.append("Expected Frequencies (under null hypothesis):")
for line in str(expected_df).split('\n'):
    report_lines.append("  " + line)
report_lines.append("")
report_lines.append("=" * 80)
report_lines.append("STATISTICAL CONCLUSION")
report_lines.append("=" * 80)
report_lines.append("")
if p_value < 0.05:
    report_lines.append(f"The chi-square test indicates a statistically significant association")
    report_lines.append(f"between comment type (outlier vs typical) and sentiment (positive vs negative)")
    report_lines.append(f"(χ²({dof}) = {chi2:.4f}, p = {p_value:.6f}).")
    report_lines.append("")
    report_lines.append("The observed difference in sentiment distribution between outliers and")
    report_lines.append("typical comments is statistically significant at the α = 0.05 level.")
    report_lines.append("")
    report_lines.append(f"Specifically, outliers showed {outlier_positive_pct:.2f}% positive sentiment")
    report_lines.append(f"compared to {typical_positive_pct:.2f}% for typical comments, a difference")
    report_lines.append(f"of {outlier_positive_pct - typical_positive_pct:.2f} percentage points.")
else:
    report_lines.append(f"The chi-square test does not indicate a statistically significant association")
    report_lines.append(f"between comment type (outlier vs typical) and sentiment (positive vs negative)")
    report_lines.append(f"(χ²({dof}) = {chi2:.4f}, p = {p_value:.6f}).")
    report_lines.append("")
    report_lines.append("The observed difference in sentiment distribution between outliers and")
    report_lines.append("typical comments is not statistically significant at the α = 0.05 level.")
    report_lines.append("")
    report_lines.append("This suggests that the sentiment distribution does not differ significantly")
    report_lines.append("between outlier and typical comments.")
report_lines.append("")
report_lines.append("=" * 80)
report_lines.append("ADDITIONAL ANALYSIS: INCLUDING NEUTRAL SENTIMENT")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append("When neutral sentiment is included in the analysis (2×3 contingency table):")
report_lines.append("")
for line in str(contingency_2x3).split('\n'):
    report_lines.append("  " + line)
report_lines.append("")
report_lines.append(f"Chi-square statistic: {chi2_3cat:.4f}")
report_lines.append(f"Degrees of freedom: {dof_3cat}")
report_lines.append(f"P-value: {p_value_3cat:.6f}")
report_lines.append(f"Effect size (Cramér's V): {cramers_v_3cat:.4f}")
report_lines.append("")
if p_value_3cat < 0.05:
    report_lines.append("With neutral sentiment included, the association is statistically significant.")
else:
    report_lines.append("With neutral sentiment included, the association is not statistically significant.")
report_lines.append("")
report_lines.append("=" * 80)
report_lines.append("RECOMMENDATIONS FOR MANUSCRIPT")
report_lines.append("=" * 80)
report_lines.append("")
if p_value < 0.05:
    report_lines.append("1. Report the chi-square test results in the Results section:")
    report_lines.append(f"   'A chi-square test for independence revealed a statistically significant")
    report_lines.append(f"   association between comment type (outlier vs typical) and sentiment")
    report_lines.append(f"   (positive vs negative), χ²({dof}) = {chi2:.4f}, p = {p_value:.6f}.")
    report_lines.append(f"   Outliers showed {outlier_positive_pct:.1f}% positive sentiment compared")
    report_lines.append(f"   to {typical_positive_pct:.1f}% for typical comments.'")
    report_lines.append("")
    report_lines.append("2. Include the contingency table in the manuscript (either in the main text")
    report_lines.append("   or as supplementary material).")
    report_lines.append("")
    report_lines.append("3. Discuss the practical significance of the finding, noting the effect size")
    report_lines.append(f"   (Cramér's V = {cramers_v:.4f}, interpreted as {effect_interpretation}).")
else:
    report_lines.append("1. Report the chi-square test results in the Results section:")
    report_lines.append(f"   'A chi-square test for independence revealed no statistically significant")
    report_lines.append(f"   association between comment type (outlier vs typical) and sentiment")
    report_lines.append(f"   (positive vs negative), χ²({dof}) = {chi2:.4f}, p = {p_value:.6f}.")
    report_lines.append(f"   Although outliers showed {outlier_positive_pct:.1f}% positive sentiment")
    report_lines.append(f"   compared to {typical_positive_pct:.1f}% for typical comments, this difference")
    report_lines.append(f"   was not statistically significant.'")
    report_lines.append("")
    report_lines.append("2. Discuss the implications: the observed difference in sentiment proportions")
    report_lines.append("   may be due to chance rather than a true association between comment type")
    report_lines.append("   and sentiment.")
report_lines.append("")
report_lines.append("3. Consider including the contingency table in supplementary materials for")
report_lines.append("   transparency.")
report_lines.append("")
report_lines.append("=" * 80)

# Save report
output_dir = os.path.join(project_root, 'outlier_analysis')
os.makedirs(output_dir, exist_ok=True)
report_path = os.path.join(output_dir, 'chi_square_sentiment_test_report.txt')

with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print(f"\n✓ Report saved to: {report_path}")

# Also save contingency tables as CSV
contingency_2x2_path = os.path.join(output_dir, 'contingency_table_2x2.csv')
contingency_2x2.to_csv(contingency_2x2_path)
print(f"✓ 2x2 Contingency table saved to: {contingency_2x2_path}")

contingency_2x3_path = os.path.join(output_dir, 'contingency_table_2x3.csv')
contingency_2x3.to_csv(contingency_2x3_path)
print(f"✓ 2x3 Contingency table saved to: {contingency_2x3_path}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
