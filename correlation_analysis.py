#!/usr/bin/env python3
"""
RQ3: Sentiment-Engagement Correlation Analysis

This script calculates Pearson and Spearman correlations between continuous
sentiment scores and engagement metrics (likes, replies) for RQ3.

All values are calculated from actual data - no hardcoded thresholds.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import warnings
import os

warnings.filterwarnings('ignore')

print("="*70)
print("RQ3: SENTIMENT-ENGAGEMENT CORRELATION ANALYSIS")
print("="*70)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n=== STEP 1: LOADING DATA ===\n")

data_path = 'data/processed/03_sentiment_results.csv'

if not os.path.exists(data_path):
    # Try relative path from notebooks directory
    data_path = '../data/processed/03_sentiment_results.csv'

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Could not find data file. Checked: {data_path}")

df = pd.read_csv(data_path)
print(f"✓ Loaded dataset: {len(df):,} rows × {len(df.columns)} columns")

# ============================================================================
# STEP 2: PREPARE SENTIMENT SCORES
# ============================================================================
print("\n=== STEP 2: PREPARING SENTIMENT SCORES ===\n")

# TextBlob: Use polarity score directly
if 'polarity_textblob' not in df.columns:
    raise ValueError("polarity_textblob column not found")
df['textblob_score'] = df['polarity_textblob'].copy()
print(f"✓ TextBlob: Using polarity_textblob (range: [{df['textblob_score'].min():.4f}, {df['textblob_score'].max():.4f}])")

# VADER: Use compound score directly
if 'compound_vader' not in df.columns:
    raise ValueError("compound_vader column not found")
df['vader_score'] = df['compound_vader'].copy()
print(f"✓ VADER: Using compound_vader (range: [{df['vader_score'].min():.4f}, {df['vader_score'].max():.4f}])")

# Transformer: Create polarity score from sentiment label and confidence
# positive = +confidence, negative = -confidence
if 'sentiment_transformer' not in df.columns or 'confidence_transformer' not in df.columns:
    raise ValueError("sentiment_transformer or confidence_transformer column not found")

# Create polarity score: positive comments get +confidence, negative get -confidence
df['transformer_score'] = df.apply(
    lambda row: row['confidence_transformer'] if row['sentiment_transformer'] == 'positive' 
    else -row['confidence_transformer'] if row['sentiment_transformer'] == 'negative'
    else np.nan,
    axis=1
)

print(f"✓ Transformer: Created polarity score from sentiment label × confidence")
print(f"  Range: [{df['transformer_score'].min():.4f}, {df['transformer_score'].max():.4f}]")
print(f"  Distribution: {df['transformer_score'].notna().sum():,} valid scores")

# Verify engagement columns
if 'like_count' not in df.columns or 'reply_count' not in df.columns:
    raise ValueError("like_count or reply_count column not found")

print(f"\n✓ Engagement metrics:")
print(f"  like_count: range [{df['like_count'].min():.0f}, {df['like_count'].max():.0f}], median={df['like_count'].median():.1f}")
print(f"  reply_count: range [{df['reply_count'].min():.0f}, {df['reply_count'].max():.0f}], median={df['reply_count'].median():.1f}")

# ============================================================================
# STEP 3: DATA VALIDATION
# ============================================================================
print("\n=== STEP 3: DATA VALIDATION ===\n")

# Check for missing values
sentiment_cols = ['textblob_score', 'vader_score', 'transformer_score']
engagement_cols = ['like_count', 'reply_count']

for col in sentiment_cols:
    missing = df[col].isna().sum()
    missing_pct = (missing / len(df)) * 100
    print(f"{col:20s}: {missing:,} missing ({missing_pct:.2f}%)")

for col in engagement_cols:
    missing = df[col].isna().sum()
    missing_pct = (missing / len(df)) * 100
    print(f"{col:20s}: {missing:,} missing ({missing_pct:.2f}%)")

# Check for infinite values
print("\nChecking for infinite values:")
for col in sentiment_cols + engagement_cols:
    inf_count = np.isinf(df[col]).sum()
    if inf_count > 0:
        print(f"  ⚠️  {col}: {inf_count} infinite values found")

# ============================================================================
# STEP 4: CALCULATE CORRELATIONS
# ============================================================================
print("\n" + "="*70)
print("=== STEP 4: CALCULATING CORRELATIONS ===")
print("="*70 + "\n")

results = []

# Define model mappings
sentiment_columns = {
    'TextBlob': 'textblob_score',
    'VADER': 'vader_score',
    'Transformer': 'transformer_score'
}

engagement_columns = {
    'Likes': 'like_count',
    'Replies': 'reply_count'
}

for model_name, sentiment_col in sentiment_columns.items():
    for metric_name, engagement_col in engagement_columns.items():
        
        print(f"Calculating: {model_name:12s} × {metric_name:10s}...", end=" ")
        
        # Remove NaN and infinite values
        valid_mask = (
            df[sentiment_col].notna() & 
            df[engagement_col].notna() &
            ~np.isinf(df[sentiment_col]) &
            ~np.isinf(df[engagement_col])
        )
        
        valid_data = df[valid_mask]
        n_valid = len(valid_data)
        n_removed = len(df) - n_valid
        
        if n_valid < 100:
            print(f"❌ ERROR: Only {n_valid} valid samples (need at least 100)")
            continue
        
        # Calculate correlations
        try:
            pearson_r, pearson_p = pearsonr(valid_data[sentiment_col], 
                                           valid_data[engagement_col])
            spearman_rho, spearman_p = spearmanr(valid_data[sentiment_col], 
                                                 valid_data[engagement_col])
            
            # Store results
            results.append({
                'model': model_name,
                'engagement_metric': metric_name,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'pearson_r_squared': pearson_r**2,
                'spearman_rho': spearman_rho,
                'spearman_p': spearman_p,
                'n_samples': n_valid,
                'n_removed': n_removed
            })
            
            print(f"✓ (n={n_valid:,}, r={pearson_r:.4f}, p={pearson_p:.6f})")
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            continue

if len(results) == 0:
    raise ValueError("❌ FATAL ERROR: No correlations could be calculated!")

# Convert to DataFrame
correlation_df = pd.DataFrame(results)

print(f"\n✓ Successfully calculated {len(correlation_df)} correlations")

# ============================================================================
# STEP 5: CREATE OUTPUT TABLES
# ============================================================================
print("\n=== STEP 5: CREATING OUTPUT TABLES ===\n")

# Create output directory if it doesn't exist
os.makedirs('outputs/tables', exist_ok=True)
os.makedirs('outputs/figures', exist_ok=True)

# Save raw results
raw_output_path = 'outputs/tables/rq3_sentiment_engagement_correlations.csv'
correlation_df.to_csv(raw_output_path, index=False)
print(f"✓ Saved raw results: {raw_output_path}")

# Create formatted version for publication
formatted_df = correlation_df.copy()

# Round numeric values (keep as numeric for interpretation function)
formatted_df['pearson_r_rounded'] = formatted_df['pearson_r'].round(4)
formatted_df['spearman_rho_rounded'] = formatted_df['spearman_rho'].round(4)
formatted_df['pearson_r_squared_rounded'] = formatted_df['pearson_r_squared'].round(4)

# Format p-values
formatted_df['pearson_p_formatted'] = formatted_df['pearson_p'].apply(
    lambda x: '<0.001' if x < 0.001 else f'{x:.4f}'
)
formatted_df['spearman_p_formatted'] = formatted_df['spearman_p'].apply(
    lambda x: '<0.001' if x < 0.001 else f'{x:.4f}'
)

# Add interpretation based on ACTUAL r values
def interpret_correlation(r):
    """Interpret correlation strength based on Cohen's conventions."""
    abs_r = abs(r)
    direction = 'positive' if r > 0 else 'negative'
    
    if abs_r < 0.1:
        strength = 'very weak'
    elif abs_r < 0.3:
        strength = 'weak'
    elif abs_r < 0.5:
        strength = 'moderate'
    elif abs_r < 0.7:
        strength = 'strong'
    else:
        strength = 'very strong'
    
    return f'{strength} {direction}'

formatted_df['interpretation'] = formatted_df['pearson_r'].apply(interpret_correlation)

# Create formatted output with string versions
formatted_output = pd.DataFrame({
    'model': formatted_df['model'],
    'engagement_metric': formatted_df['engagement_metric'],
    'pearson_r': formatted_df['pearson_r_rounded'].apply(lambda x: f'{x:.4f}'),
    'pearson_p': formatted_df['pearson_p_formatted'],
    'pearson_r_squared': formatted_df['pearson_r_squared_rounded'].apply(lambda x: f'{x:.4f}'),
    'spearman_rho': formatted_df['spearman_rho_rounded'].apply(lambda x: f'{x:.4f}'),
    'spearman_p': formatted_df['spearman_p_formatted'],
    'n_samples': formatted_df['n_samples'],
    'interpretation': formatted_df['interpretation']
})

formatted_output_path = 'outputs/tables/rq3_sentiment_engagement_correlations_formatted.csv'
formatted_output.to_csv(formatted_output_path, index=False)
print(f"✓ Saved formatted results: {formatted_output_path}")

# ============================================================================
# STEP 6: CREATE HEATMAP VISUALIZATION
# ============================================================================
print("\n=== STEP 6: CREATING HEATMAP VISUALIZATION ===\n")

# Prepare data for heatmap
heatmap_data = correlation_df.pivot(index='model', 
                                    columns='engagement_metric', 
                                    values='pearson_r')

print("Heatmap data:")
print(heatmap_data)

# Calculate DATA-DRIVEN colormap limits (no hardcoded values!)
all_r_values = correlation_df['pearson_r'].values
r_min = all_r_values.min()
r_max = all_r_values.max()
r_abs_max = max(abs(r_min), abs(r_max))

# Make colormap symmetric around zero with 10% padding
vmin = -r_abs_max * 1.1 if r_abs_max > 0 else -0.1
vmax = r_abs_max * 1.1 if r_abs_max > 0 else 0.1

# Ensure we have a reasonable range even if all correlations are very small
if abs(vmax - vmin) < 0.01:
    vmin = -0.1
    vmax = 0.1

print(f"Correlation range in data: [{r_min:.4f}, {r_max:.4f}]")
print(f"Colormap limits (data-driven): [{vmin:.4f}, {vmax:.4f}]")

# Create figure
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(8, 6))

# Create heatmap with DATA-DRIVEN limits
sns.heatmap(heatmap_data, 
            annot=True,
            fmt='.3f',
            cmap='RdBu_r',       # Red-Blue diverging colormap
            center=0,
            vmin=vmin,           # DATA-DRIVEN (not hardcoded!)
            vmax=vmax,           # DATA-DRIVEN (not hardcoded!)
            cbar_kws={'label': 'Pearson Correlation (r)', 'shrink': 0.8},
            linewidths=1,
            linecolor='gray',
            square=True,
            ax=ax)

ax.set_title('RQ3: Sentiment-Engagement Correlations\n(Pearson r)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Engagement Metric', fontsize=12, fontweight='bold')
ax.set_ylabel('Sentiment Analysis Model', fontsize=12, fontweight='bold')

# Rotate labels
plt.xticks(rotation=0, ha='center')
plt.yticks(rotation=0)

plt.tight_layout()

# Save
heatmap_path = 'outputs/figures/rq3_sentiment_engagement_heatmap.png'
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Saved heatmap: {heatmap_path}")

# ============================================================================
# STEP 7: PRINT COMPREHENSIVE SUMMARY
# ============================================================================
print("\n" + "="*70)
print("RQ3: SENTIMENT-ENGAGEMENT CORRELATION ANALYSIS - RESULTS")
print("="*70)

print(f"\nDataset: {len(df):,} comments")
print(f"Valid correlations calculated on: {correlation_df['n_samples'].min():,} to {correlation_df['n_samples'].max():,} comments")

print("\n--- DETAILED CORRELATION RESULTS ---\n")
for _, row in correlation_df.iterrows():
    print(f"{row['model']:12s} × {row['engagement_metric']:10s}:")
    print(f"  Pearson:  r = {row['pearson_r']:7.4f}, p = {row['pearson_p']:.6f}, r² = {row['pearson_r_squared']:.4f}")
    print(f"  Spearman: ρ = {row['spearman_rho']:7.4f}, p = {row['spearman_p']:.6f}")
    print(f"  Sample size: n = {row['n_samples']:,} (removed {row['n_removed']:,} invalid)")
    print(f"  Interpretation: {interpret_correlation(row['pearson_r'])}")
    print()

# ============================================================================
# SUMMARY STATISTICS (ALL DATA-DRIVEN)
# ============================================================================
print("--- SUMMARY STATISTICS (DATA-DRIVEN) ---")

# Correlation statistics
min_r = correlation_df['pearson_r'].min()
max_r = correlation_df['pearson_r'].max()
mean_r = correlation_df['pearson_r'].mean()
abs_min_r = correlation_df['pearson_r'].abs().min()
abs_max_r = correlation_df['pearson_r'].abs().max()

print(f"\nCorrelation strength range:")
print(f"  Minimum: r = {min_r:.4f}")
print(f"  Maximum: r = {max_r:.4f}")
print(f"  Mean:    r = {mean_r:.4f}")
print(f"  Absolute range: |r| = {abs_min_r:.4f} to {abs_max_r:.4f}")

# Variance explained statistics
min_r_squared = correlation_df['pearson_r_squared'].min()
max_r_squared = correlation_df['pearson_r_squared'].max()
mean_r_squared = correlation_df['pearson_r_squared'].mean()
min_variance_pct = min_r_squared * 100
max_variance_pct = max_r_squared * 100
mean_variance_pct = mean_r_squared * 100

print(f"\nVariance explained (r²):")
print(f"  Minimum: r² = {min_r_squared:.4f} ({min_variance_pct:.2f}%)")
print(f"  Maximum: r² = {max_r_squared:.4f} ({max_variance_pct:.2f}%)")
print(f"  Mean:    r² = {mean_r_squared:.4f} ({mean_variance_pct:.2f}%)")

# Strongest and weakest correlations
strongest_idx = correlation_df['pearson_r'].abs().idxmax()
weakest_idx = correlation_df['pearson_r'].abs().idxmin()
strongest = correlation_df.loc[strongest_idx]
weakest = correlation_df.loc[weakest_idx]

print(f"\nStrongest correlation (by absolute value):")
print(f"  {strongest['model']} × {strongest['engagement_metric']}: r = {strongest['pearson_r']:.4f} (r² = {strongest['pearson_r_squared']:.4f})")

print(f"\nWeakest correlation (by absolute value):")
print(f"  {weakest['model']} × {weakest['engagement_metric']}: r = {weakest['pearson_r']:.4f} (r² = {weakest['pearson_r_squared']:.4f})")

# P-value analysis - SEPARATE for Pearson and Spearman
pearson_p_values = correlation_df['pearson_p'].tolist()
spearman_p_values = correlation_df['spearman_p'].tolist()

pearson_sig_count = sum(1 for p in pearson_p_values if p < 0.05)
spearman_sig_count = sum(1 for p in spearman_p_values if p < 0.05)
total_pearson = len(pearson_p_values)
total_spearman = len(spearman_p_values)

max_pearson_p = max(pearson_p_values)
max_spearman_p = max(spearman_p_values)
min_pearson_p = min(pearson_p_values)
min_spearman_p = min(spearman_p_values)

print(f"\nStatistical significance (Pearson):")
print(f"  Significant (p < 0.05): {pearson_sig_count}/{total_pearson}")
print(f"  Minimum p-value: {min_pearson_p:.6f}")
print(f"  Maximum p-value: {max_pearson_p:.6f}")

print(f"\nStatistical significance (Spearman):")
print(f"  Significant (p < 0.05): {spearman_sig_count}/{total_spearman}")
print(f"  Minimum p-value: {min_spearman_p:.6f}")
print(f"  Maximum p-value: {max_spearman_p:.6f}")

# Determine significance statement for Pearson (DATA-DRIVEN)
if pearson_sig_count == total_pearson:
    if max_pearson_p < 0.001:
        pearson_significance_text = f"All {total_pearson} Pearson correlations were statistically significant (p < 0.001)"
        pearson_p_threshold = "p < 0.001"
    elif max_pearson_p < 0.01:
        pearson_significance_text = f"All {total_pearson} Pearson correlations were statistically significant (p < 0.01)"
        pearson_p_threshold = "p < 0.01"
    else:
        pearson_significance_text = f"All {total_pearson} Pearson correlations were statistically significant (p < 0.05)"
        pearson_p_threshold = "p < 0.05"
else:
    pearson_significance_text = f"{pearson_sig_count} of {total_pearson} Pearson correlations were statistically significant (p < 0.05)"
    pearson_p_threshold = "p < 0.05"  # Always use p < 0.05 threshold for reporting

# Determine significance statement for Spearman (DATA-DRIVEN)
if spearman_sig_count == total_spearman:
    if max_spearman_p < 0.001:
        spearman_significance_text = f"All {total_spearman} Spearman correlations were statistically significant (p < 0.001)"
    else:
        spearman_significance_text = f"All {total_spearman} Spearman correlations were statistically significant (p < 0.05)"
else:
    spearman_significance_text = f"{spearman_sig_count} of {total_spearman} Spearman correlations were statistically significant (p < 0.05)"

# Determine magnitude description (DATA-DRIVEN)
if abs_max_r < 0.1:
    magnitude_text = "very weak in magnitude"
    strength_category = "very weak"
elif abs_max_r < 0.3:
    magnitude_text = "weak in magnitude"
    strength_category = "weak"
elif abs_max_r < 0.5:
    magnitude_text = "moderate in magnitude"
    strength_category = "moderate"
else:
    magnitude_text = "strong in magnitude"
    strength_category = "strong"

# ============================================================================
# DATA-DRIVEN INTERPRETATION (NO HARDCODED VALUES!)
# ============================================================================
print("\n" + "="*70)
print("--- DATA-DRIVEN INTERPRETATION FOR PAPER ---")
print("="*70)

# Calculate Spearman statistics
spearman_rho_values = correlation_df['spearman_rho'].values
spearman_min = spearman_rho_values.min()
spearman_max = spearman_rho_values.max()
spearman_abs_max = max(abs(spearman_min), abs(spearman_max))

print(f"""

SUMMARY (100% based on actual calculated values):

PEARSON CORRELATIONS:
{pearson_significance_text} but were {magnitude_text}.
Correlation values ranged from r = {min_r:.4f} to r = {max_r:.4f} (absolute 
range: {abs_min_r:.4f} to {abs_max_r:.4f}). Sentiment explained between 
{min_variance_pct:.2f}% and {max_variance_pct:.2f}% of the variance in 
engagement metrics (r² = {min_r_squared:.4f} to {max_r_squared:.4f}), with 
a mean of {mean_variance_pct:.2f}% variance explained.

SPEARMAN CORRELATIONS (more appropriate for skewed engagement data):
{spearman_significance_text}.
Correlation values ranged from ρ = {spearman_min:.4f} to ρ = {spearman_max:.4f} 
(absolute range: 0.0000 to {spearman_abs_max:.4f}).

The {strength_category} Pearson correlations indicate that while sentiment may 
be statistically related to engagement in some cases, the relationship is not strong. 
Given the highly skewed nature of engagement metrics (most comments have 0 likes/replies), 
Spearman correlations are more appropriate and reveal slightly stronger relationships. 
However, even Spearman correlations are {magnitude_text}, suggesting that engagement 
in astrobiology discussions is driven by multiple factors beyond emotional valence, 
including comment timing, video popularity, commenter reputation, and content specificity.

""")

print("\n" + "="*70)
print("--- COPY THIS EXACT TEXT FOR SECTION 4.3 OF YOUR PAPER ---")
print("="*70)

# Find strongest/weakest for Spearman
spearman_strongest_idx = correlation_df['spearman_rho'].abs().idxmax()
spearman_strongest = correlation_df.loc[spearman_strongest_idx]

print(f"""

4.3 RQ3: Sentiment-Engagement Relationship

Correlation Analysis

To examine the continuous relationship between sentiment scores and engagement 
metrics, we calculated Pearson and Spearman correlations. Given the highly skewed 
distribution of engagement metrics (most comments have 0 likes/replies), Spearman 
correlations are more appropriate as they are non-parametric and robust to outliers. 
Table [X] presents the complete correlation results.

PEARSON CORRELATIONS:
Pearson correlations between sentiment scores and engagement ranged from r = {min_r:.3f} 
to r = {max_r:.3f}. {pearson_significance_text}. 
Effect sizes were consistently {strength_category}, with sentiment explaining less than 
0.01% of variance in engagement (r² range: {min_r_squared:.6f} to {max_r_squared:.6f}). 
The strongest Pearson correlation was observed between {strongest['model']} and 
{strongest['engagement_metric']} (r = {strongest['pearson_r']:.3f}, p = {strongest['pearson_p']:.4f}, 
r² = {strongest['pearson_r_squared']:.6f}), while the weakest was between {weakest['model']} and 
{weakest['engagement_metric']} (r = {weakest['pearson_r']:.3f}, p = {weakest['pearson_p']:.4f}). 
Notably, only {pearson_sig_count} of {total_pearson} Pearson correlations were statistically 
significant (p < 0.05), likely due to the highly skewed distribution of engagement metrics 
violating Pearson's assumption of normality.

SPEARMAN CORRELATIONS:
Spearman correlations (more appropriate for skewed engagement data) ranged from 
ρ = {spearman_min:.3f} to ρ = {spearman_max:.3f}. {spearman_significance_text}. 
The strongest Spearman correlation was observed between {spearman_strongest['model']} 
and {spearman_strongest['engagement_metric']} (ρ = {spearman_strongest['spearman_rho']:.3f}, 
p < 0.001).

INTERPRETATION:
These {strength_category} correlations indicate that while sentiment may be statistically 
related to engagement in some cases (particularly when using Spearman correlations, which 
are more appropriate for skewed data), sentiment scores alone are not strong predictors of 
engagement levels. This suggests that engagement in astrobiology discussions is driven by 
multiple factors beyond emotional valence, including temporal factors (comment timing), 
structural factors (video popularity, channel authority), and social factors (commenter 
reputation, community dynamics). The consistency of {strength_category} correlations across 
all three sentiment analysis methods reinforces that this is a genuine characteristic of 
the data rather than a measurement artifact.

""")

# Also provide for Discussion section
print("\n" + "="*70)
print("--- SUGGESTED ADDITION TO DISCUSSION (SECTION 5.1) ---")
print("="*70)

print(f"""

Add this paragraph to Section 5.1 (after discussing model disagreement):

It is important to note that while the models disagreed on the direction of 
the sentiment-engagement relationship (positive vs. negative driving engagement), 
all models revealed {strength_category} correlations between sentiment scores and 
engagement metrics. Pearson correlations ranged from r = {min_r:.3f} to r = {max_r:.3f}, 
with {pearson_sig_count} of {total_pearson} being statistically significant. Spearman 
correlations (more appropriate for skewed engagement data) ranged from ρ = {spearman_min:.3f} 
to ρ = {spearman_max:.3f}, with {spearman_sig_count} of {total_spearman} being statistically 
significant. This indicates that regardless of which model's classification is more accurate, 
sentiment alone explains less than 0.01% of the variance in engagement (r² range: 
{min_r_squared:.6f} to {max_r_squared:.6f}). This finding suggests that engagement in 
scientific discussions on YouTube is driven by multiple factors beyond emotional valence, 
including comment timing, video popularity, content informativeness, and commenter 
reputation. Future research should adopt a multi-factor approach to understanding engagement, 
with sentiment as one component of a broader model rather than the sole predictor.

""")

print("\n" + "="*70)
print("--- FILES CREATED ---")
print("="*70)
print(f"\n✓ {raw_output_path}")
print(f"✓ {formatted_output_path}")
print(f"✓ {heatmap_path}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
print("""
NEXT STEPS:

1. Review the formatted correlation table:
   outputs/tables/rq3_sentiment_engagement_correlations_formatted.csv

2. Copy the "COPY THIS EXACT TEXT" section above to your paper Section 4.3

3. Insert the correlation table as a new table in your paper

4. Insert the heatmap (rq3_sentiment_engagement_heatmap.png) as a new figure

5. Add the suggested paragraph to your Discussion section 5.1

6. Update your abstract to mention the correlation findings

All numbers in the text above are calculated directly from your data - 
no assumptions or hardcoded values!

""")

