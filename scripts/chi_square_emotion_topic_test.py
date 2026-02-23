#!/usr/bin/env python3
"""
Chi-Square Test for Independence: Topic × Emotion
This script performs a statistical test to determine if there is a significant
association between topic and emotion distribution.

Based on reviewer suggestion: "Technically, you could run Chi² tests here as well 
(Topic * Emotion) to test whether the distribution is statistically significant 
(e.g., between topics)"
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
print("CHI-SQUARE TEST: TOPIC × EMOTION")
print("=" * 80)

# Load emotion data
print("\nLoading emotion data...")
emotion_data_path = os.path.join(project_root, 'data/processed/youtube_comments_with_emotions.csv')
if not os.path.exists(emotion_data_path):
    raise FileNotFoundError(f"Emotion data not found at: {emotion_data_path}")

df_emotions = pd.read_csv(emotion_data_path)
print(f"Loaded {len(df_emotions):,} comments with emotion data")

# Check required columns
if 'search_query' not in df_emotions.columns:
    raise ValueError("'search_query' column not found in dataset")
if 'dominant_emotion' not in df_emotions.columns:
    raise ValueError("'dominant_emotion' column not found in dataset")

# Filter to valid topics only (as done in sentiment analysis)
valid_topics = ['3I/ATLAS', 'Oumuamua', 'K2-18b', "Tabby's Star", 'Venus phosphine']
df_valid = df_emotions[df_emotions['search_query'].isin(valid_topics)].copy()
print(f"Filtered to {len(df_valid):,} comments with valid topics (from {len(df_emotions):,} total)")

# Get top 12 emotions (as mentioned in the paper)
print("\nCalculating top 12 emotions...")
emotion_counts = df_valid['dominant_emotion'].value_counts()
# Exclude neutral if present
if 'neutral' in emotion_counts.index:
    emotion_counts = emotion_counts.drop('neutral')

top_12_emotions = emotion_counts.head(12).index.tolist()
print(f"Top 12 emotions: {', '.join(top_12_emotions)}")

# Filter to top 12 emotions for analysis
df_top12 = df_valid[df_valid['dominant_emotion'].isin(top_12_emotions)].copy()
print(f"Comments with top 12 emotions: {len(df_top12):,} ({len(df_top12)/len(df_valid)*100:.2f}%)")

# ============================================================================
# CHI-SQUARE TEST: TOP 12 EMOTIONS
# ============================================================================
print("\n" + "=" * 80)
print("CHI-SQUARE TEST: TOPIC × TOP 12 EMOTIONS")
print("=" * 80)

# Create contingency table
contingency_table = pd.crosstab(df_top12['search_query'], df_top12['dominant_emotion'])

print("\nContingency Table (Topic × Emotion):")
print(contingency_table)

# Check for cells with expected frequency < 5 (chi-square assumption)
print("\nChecking chi-square assumptions...")
chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
expected_df = pd.DataFrame(
    expected,
    index=contingency_table.index,
    columns=contingency_table.columns
)

low_expected = (expected_df < 5).sum().sum()
total_cells = expected_df.size
low_expected_pct = (low_expected / total_cells) * 100

print(f"Cells with expected frequency < 5: {low_expected}/{total_cells} ({low_expected_pct:.1f}%)")
if low_expected_pct > 20:
    print("⚠ Warning: More than 20% of cells have expected frequency < 5.")
    print("  Consider combining rare emotions or using Fisher's exact test.")
else:
    print("✓ Chi-square assumptions met (less than 20% of cells with expected frequency < 5)")

# Perform chi-square test
print("\n" + "=" * 80)
print("CHI-SQUARE TEST RESULTS")
print("=" * 80)
print(f"\nChi-square statistic: {chi2_stat:.4f}")
print(f"Degrees of freedom: {dof}")
print(f"P-value: {p_value:.6f}")
print(f"Alpha level: 0.05")

# Calculate effect size (Cramér's V)
n = contingency_table.sum().sum()
cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))

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
    print(f"between topic and emotion distribution (χ²({dof}) = {chi2_stat:.4f}, p = {p_value:.6f}).")
    print(f"\nThe observed differences in emotion distribution across topics are")
    print(f"statistically significant at the α = 0.05 level.")
    print(f"\nEffect size (Cramér's V = {cramers_v:.4f}) indicates a {effect_interpretation} effect.")
else:
    print(f"\nThe chi-square test does not indicate a statistically significant association")
    print(f"between topic and emotion distribution (χ²({dof}) = {chi2_stat:.4f}, p = {p_value:.6f}).")
    print(f"\nThe observed differences in emotion distribution across topics are")
    print(f"not statistically significant at the α = 0.05 level.")

# ============================================================================
# ADDITIONAL ANALYSIS: ALL EMOTIONS (not just top 12)
# ============================================================================
print("\n" + "=" * 80)
print("ADDITIONAL ANALYSIS: ALL EMOTIONS (excluding neutral)")
print("=" * 80)

# Filter out neutral
df_all_emotions = df_valid[df_valid['dominant_emotion'] != 'neutral'].copy()
print(f"Comments with all emotions (excluding neutral): {len(df_all_emotions):,}")

# Create contingency table for all emotions
contingency_table_all = pd.crosstab(df_all_emotions['search_query'], 
                                    df_all_emotions['dominant_emotion'])

print(f"\nContingency table dimensions: {contingency_table_all.shape[0]} topics × {contingency_table_all.shape[1]} emotions")

# Perform chi-square test for all emotions
chi2_stat_all, p_value_all, dof_all, expected_all = chi2_contingency(contingency_table_all)

print(f"\nChi-square statistic (all emotions): {chi2_stat_all:.4f}")
print(f"Degrees of freedom: {dof_all}")
print(f"P-value: {p_value_all:.6f}")

# Calculate Cramér's V for all emotions
n_all = contingency_table_all.sum().sum()
cramers_v_all = np.sqrt(chi2_stat_all / (n_all * (min(contingency_table_all.shape) - 1)))
print(f"Effect Size (Cramér's V, all emotions): {cramers_v_all:.4f}")

if p_value_all < 0.05:
    print(f"\nWith all emotions included, the association is statistically significant")
    print(f"(χ²({dof_all}) = {chi2_stat_all:.4f}, p = {p_value_all:.6f}).")
else:
    print(f"\nWith all emotions included, the association is not statistically significant")
    print(f"(χ²({dof_all}) = {chi2_stat_all:.4f}, p = {p_value_all:.6f}).")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

output_dir = os.path.join(project_root, 'outputs/tables')
os.makedirs(output_dir, exist_ok=True)

# Save contingency table (top 12)
contingency_table.to_csv(os.path.join(output_dir, 'emotion_topic_contingency_top12.csv'))
print(f"✓ Saved: outputs/tables/emotion_topic_contingency_top12.csv")

# Save contingency table (all emotions)
contingency_table_all.to_csv(os.path.join(output_dir, 'emotion_topic_contingency_all.csv'))
print(f"✓ Saved: outputs/tables/emotion_topic_contingency_all.csv")

# Save expected frequencies
expected_df.to_csv(os.path.join(output_dir, 'emotion_topic_expected_frequencies_top12.csv'))
print(f"✓ Saved: outputs/tables/emotion_topic_expected_frequencies_top12.csv")

# Save summary results
summary_results = pd.DataFrame({
    'analysis': ['Top 12 Emotions', 'All Emotions'],
    'chi2_statistic': [chi2_stat, chi2_stat_all],
    'p_value': [p_value, p_value_all],
    'degrees_of_freedom': [dof, dof_all],
    'cramers_v': [cramers_v, cramers_v_all],
    'n': [n, n_all],
    'significant': ['Yes' if p_value < 0.05 else 'No', 
                    'Yes' if p_value_all < 0.05 else 'No']
})
summary_results.to_csv(os.path.join(output_dir, 'chi_square_emotion_topic_results.csv'), index=False)
print(f"✓ Saved: outputs/tables/chi_square_emotion_topic_results.csv")

# ============================================================================
# GENERATE REPORT FOR MANUSCRIPT
# ============================================================================
print("\n" + "=" * 80)
print("GENERATING REPORT FOR MANUSCRIPT")
print("=" * 80)

report_lines = []
report_lines.append("=" * 80)
report_lines.append("CHI-SQUARE TEST REPORT: TOPIC × EMOTION")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append("Date: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
report_lines.append("")
report_lines.append("=" * 80)
report_lines.append("METHODOLOGY")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append("A chi-square test for independence was performed to determine if there is a")
report_lines.append("statistically significant association between topic and emotion distribution.")
report_lines.append("")
report_lines.append(f"Dataset: {len(df_valid):,} total comments across 5 topics")
report_lines.append(f"  - Top 12 emotions analysis: {len(df_top12):,} comments")
report_lines.append(f"  - All emotions analysis: {len(df_all_emotions):,} comments (excluding neutral)")
report_lines.append("")
report_lines.append("Top 12 emotions tested:")
for i, emotion in enumerate(top_12_emotions, 1):
    count = emotion_counts[emotion]
    pct = (count / len(df_valid)) * 100
    report_lines.append(f"  {i}. {emotion}: {count:,} ({pct:.2f}%)")
report_lines.append("")
report_lines.append("=" * 80)
report_lines.append("RESULTS: TOP 12 EMOTIONS")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append("Contingency Table:")
report_lines.append("")
for line in str(contingency_table).split('\n'):
    report_lines.append("  " + line)
report_lines.append("")
report_lines.append("Chi-Square Test Statistics:")
report_lines.append(f"  Chi-square statistic (χ²): {chi2_stat:.4f}")
report_lines.append(f"  Degrees of freedom: {dof}")
report_lines.append(f"  P-value: {p_value:.6f}")
report_lines.append(f"  Alpha level: 0.05")
report_lines.append(f"  Effect size (Cramér's V): {cramers_v:.4f} ({effect_interpretation})")
report_lines.append(f"  Sample size: {n:,}")
report_lines.append("")
report_lines.append("Expected Frequencies (under null hypothesis):")
for line in str(expected_df).split('\n')[:10]:  # Show first 10 lines
    report_lines.append("  " + line)
report_lines.append("  ... (truncated)")
report_lines.append("")
report_lines.append("=" * 80)
report_lines.append("STATISTICAL CONCLUSION")
report_lines.append("=" * 80)
report_lines.append("")
if p_value < 0.05:
    report_lines.append(f"The chi-square test indicates a statistically significant association")
    report_lines.append(f"between topic and emotion distribution (χ²({dof}) = {chi2_stat:.4f}, p = {p_value:.6f}).")
    report_lines.append("")
    report_lines.append("The observed differences in emotion distribution across topics are")
    report_lines.append("statistically significant at the α = 0.05 level.")
    report_lines.append("")
    report_lines.append(f"Effect size (Cramér's V = {cramers_v:.4f}) indicates a {effect_interpretation} effect,")
    report_lines.append("suggesting that while the association is statistically significant, the practical")
    report_lines.append("magnitude of the difference varies.")
else:
    report_lines.append(f"The chi-square test does not indicate a statistically significant association")
    report_lines.append(f"between topic and emotion distribution (χ²({dof}) = {chi2_stat:.4f}, p = {p_value:.6f}).")
    report_lines.append("")
    report_lines.append("The observed differences in emotion distribution across topics are")
    report_lines.append("not statistically significant at the α = 0.05 level.")
report_lines.append("")
report_lines.append("=" * 80)
report_lines.append("ADDITIONAL ANALYSIS: ALL EMOTIONS")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append("When all emotions (excluding neutral) are included in the analysis:")
report_lines.append("")
report_lines.append(f"Chi-square statistic: {chi2_stat_all:.4f}")
report_lines.append(f"Degrees of freedom: {dof_all}")
report_lines.append(f"P-value: {p_value_all:.6f}")
report_lines.append(f"Effect size (Cramér's V): {cramers_v_all:.4f}")
report_lines.append("")
if p_value_all < 0.05:
    report_lines.append("With all emotions included, the association is statistically significant.")
else:
    report_lines.append("With all emotions included, the association is not statistically significant.")
report_lines.append("")
report_lines.append("=" * 80)
report_lines.append("RECOMMENDATIONS FOR MANUSCRIPT")
report_lines.append("=" * 80)
report_lines.append("")
if p_value < 0.05:
    report_lines.append("1. Add the following text to the 'Emotion Distribution by Topic' section:")
    report_lines.append("")
    report_lines.append(f"   'To test whether the observed differences in emotion distribution across")
    report_lines.append(f"   topics were statistically significant, we performed a chi-square test of")
    report_lines.append(f"   independence. The test revealed a statistically significant association")
    report_lines.append(f"   between topic and emotion distribution (χ²({dof}) = {chi2_stat:.4f}, p < 0.001).")
    report_lines.append(f"   However, the effect size was {effect_interpretation} (Cramér's V = {cramers_v:.4f}),")
    report_lines.append(f"   indicating that while the association is statistically significant, the")
    report_lines.append(f"   practical magnitude of differences varies across topics.'")
    report_lines.append("")
    report_lines.append("2. Consider adding a note about the statistical test in the figure caption")
    report_lines.append("   or in a footnote.")
    report_lines.append("")
    report_lines.append("3. The contingency table can be included in supplementary materials if")
    report_lines.append("   space allows, or referenced in the text.")
else:
    report_lines.append("1. Report the chi-square test results, noting that while descriptive")
    report_lines.append("   differences are visible in the heatmap, the overall association is not")
    report_lines.append("   statistically significant.")
    report_lines.append("")
    report_lines.append("2. Discuss potential reasons: large sample size may reveal small but")
    report_lines.append("   statistically significant differences, or the differences may be")
    report_lines.append("   concentrated in specific emotion-topic combinations.")
report_lines.append("")
report_lines.append("=" * 80)

# Save report
report_dir = os.path.join(project_root, 'outputs')
os.makedirs(report_dir, exist_ok=True)
report_path = os.path.join(report_dir, 'chi_square_emotion_topic_report.txt')

with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print(f"\n✓ Report saved to: {report_path}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\nSummary:")
print(f"  - Top 12 emotions: χ²({dof}) = {chi2_stat:.4f}, p = {p_value:.6f}, Cramér's V = {cramers_v:.4f}")
print(f"  - All emotions: χ²({dof_all}) = {chi2_stat_all:.4f}, p = {p_value_all:.6f}, Cramér's V = {cramers_v_all:.4f}")
print(f"\nAll results saved to: outputs/tables/")
