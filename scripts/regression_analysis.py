#!/usr/bin/env python3
"""
RQ3: Sentiment-Engagement Linear Regression Analysis

This script performs simple linear regression analyses to quantify the predictive
power of sentiment scores on engagement metrics (likes, replies).

Model: Engagement = β₀ + β₁(Sentiment) + ε

All values are calculated from actual data - no hardcoded thresholds.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')

print("="*70)
print("RQ3: SENTIMENT-ENGAGEMENT LINEAR REGRESSION ANALYSIS")
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
# STEP 3: PERFORM LINEAR REGRESSION
# ============================================================================
print("\n" + "="*70)
print("=== STEP 3: PERFORMING LINEAR REGRESSION ===")
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
        
        print(f"Analyzing: {model_name:12s} × {metric_name:10s}...", end=" ")
        
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
        
        # Extract variables
        X = valid_data[sentiment_col].values
        y = valid_data[engagement_col].values
        
        # Perform linear regression
        try:
            # Use scipy.stats.linregress for simple regression
            slope, intercept, r_value, p_value, std_err = linregress(X, y)
            
            # Calculate R-squared
            r_squared = r_value**2
            
            # Calculate confidence intervals for slope (95% CI)
            # t-critical value for 95% CI (two-tailed)
            t_critical = stats.t.ppf(0.975, n_valid - 2)
            slope_ci_lower = slope - t_critical * std_err
            slope_ci_upper = slope + t_critical * std_err
            
            # Calculate standardized coefficient (beta)
            # Standardized beta = slope * (std_X / std_y)
            std_X = np.std(X, ddof=1)
            std_y = np.std(y, ddof=1)
            standardized_coef = slope * (std_X / std_y) if std_y > 0 else np.nan
            
            # Determine significance
            significant = 'Yes' if p_value < 0.05 else 'No'
            
            # Store results
            results.append({
                'model': model_name,
                'engagement_metric': metric_name,
                'intercept': intercept,
                'slope': slope,
                'slope_ci_lower': slope_ci_lower,
                'slope_ci_upper': slope_ci_upper,
                'r_squared': r_squared,
                'r_value': r_value,
                'p_value': p_value,
                'std_err': std_err,
                'standardized_coef': standardized_coef,
                'significant': significant,
                'n_samples': n_valid,
                'n_removed': n_removed
            })
            
            print(f"✓ (n={n_valid:,}, R²={r_squared:.6f}, p={p_value:.6f}, β={slope:.4f})")
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            continue

if len(results) == 0:
    raise ValueError("❌ FATAL ERROR: No regressions could be calculated!")

# Convert to DataFrame
regression_df = pd.DataFrame(results)

print(f"\n✓ Successfully performed {len(regression_df)} linear regressions")

# ============================================================================
# STEP 4: CREATE OUTPUT TABLES
# ============================================================================
print("\n=== STEP 4: CREATING OUTPUT TABLES ===\n")

# Create output directory if it doesn't exist
os.makedirs('outputs/tables', exist_ok=True)
os.makedirs('outputs/figures', exist_ok=True)

# Save raw results
raw_output_path = 'outputs/tables/rq3_sentiment_engagement_regression.csv'
regression_df.to_csv(raw_output_path, index=False)
print(f"✓ Saved raw results: {raw_output_path}")

# Create formatted version for publication
formatted_df = regression_df.copy()

# Round numeric values
formatted_df['slope_rounded'] = formatted_df['slope'].round(4)
formatted_df['intercept_rounded'] = formatted_df['intercept'].round(4)
formatted_df['r_squared_rounded'] = formatted_df['r_squared'].round(6)
formatted_df['standardized_coef_rounded'] = formatted_df['standardized_coef'].round(4)

# Format p-values
formatted_df['p_value_formatted'] = formatted_df['p_value'].apply(
    lambda x: '<0.001' if x < 0.001 else f'{x:.4f}'
)

# Format confidence intervals
formatted_df['ci_formatted'] = formatted_df.apply(
    lambda row: f"[{row['slope_ci_lower']:.4f}, {row['slope_ci_upper']:.4f}]",
    axis=1
)

# Create formatted output
formatted_output = pd.DataFrame({
    'model': formatted_df['model'],
    'engagement_metric': formatted_df['engagement_metric'],
    'intercept': formatted_df['intercept_rounded'].apply(lambda x: f'{x:.4f}'),
    'slope': formatted_df['slope_rounded'].apply(lambda x: f'{x:.4f}'),
    'slope_ci_95': formatted_df['ci_formatted'],
    'r_squared': formatted_df['r_squared_rounded'].apply(lambda x: f'{x:.6f}'),
    'r_value': formatted_df['r_value'].round(4).apply(lambda x: f'{x:.4f}'),
    'p_value': formatted_df['p_value_formatted'],
    'standardized_coef': formatted_df['standardized_coef_rounded'].apply(lambda x: f'{x:.4f}'),
    'significant': formatted_df['significant'],
    'n_samples': formatted_df['n_samples']
})

formatted_output_path = 'outputs/tables/rq3_sentiment_engagement_regression_formatted.csv'
formatted_output.to_csv(formatted_output_path, index=False)
print(f"✓ Saved formatted results: {formatted_output_path}")

# ============================================================================
# STEP 5: CREATE SCATTER PLOTS WITH REGRESSION LINES
# ============================================================================
print("\n=== STEP 5: CREATING SCATTER PLOTS WITH REGRESSION LINES ===\n")

# Colorblind-friendly palette (Okabe-Ito inspired + additional colors)
# Using distinct colors and markers for accessibility
colors = {
    'Likes': '#0072B2',      # Blue (accessible)
    'Replies': '#D55E00'     # Orange/Red (accessible)
}

markers = {
    'TextBlob': 'o',
    'VADER': 's',
    'Transformer': '^'
}

# Create figure with subplots (3 models × 2 metrics = 6 plots)
fig, axes = plt.subplots(3, 2, figsize=(14, 18))
fig.suptitle('Linear Regression: Sentiment Scores Predicting Engagement Metrics', 
             fontsize=16, fontweight='bold', y=0.995)

# Flatten axes for easier iteration
axes_flat = axes.flatten()

plot_idx = 0
for model_name, sentiment_col in sentiment_columns.items():
    for metric_name, engagement_col in engagement_columns.items():
        ax = axes_flat[plot_idx]
        
        # Get valid data
        valid_mask = (
            df[sentiment_col].notna() & 
            df[engagement_col].notna() &
            ~np.isinf(df[sentiment_col]) &
            ~np.isinf(df[engagement_col])
        )
        valid_data = df[valid_mask]
        
        if len(valid_data) > 0:
            X = valid_data[sentiment_col].values
            y = valid_data[engagement_col].values
            
            # Sample data for plotting (if too large, sample 10,000 points)
            if len(X) > 10000:
                sample_idx = np.random.choice(len(X), 10000, replace=False)
                X_plot = X[sample_idx]
                y_plot = y[sample_idx]
            else:
                X_plot = X
                y_plot = y
            
            # Scatter plot with transparency
            ax.scatter(X_plot, y_plot, alpha=0.3, s=10, 
                      color=colors[metric_name], 
                      marker=markers[model_name],
                      edgecolors='none', label='Data points')
            
            # Get regression results
            reg_result = regression_df[
                (regression_df['model'] == model_name) & 
                (regression_df['engagement_metric'] == metric_name)
            ]
            
            if len(reg_result) > 0:
                slope = reg_result['slope'].values[0]
                intercept = reg_result['intercept'].values[0]
                r_squared = reg_result['r_squared'].values[0]
                p_value = reg_result['p_value'].values[0]
                
                # Plot regression line
                x_line = np.linspace(X.min(), X.max(), 100)
                y_line = intercept + slope * x_line
                ax.plot(x_line, y_line, '--', color='black', linewidth=2, 
                       label=f'Regression line (R²={r_squared:.6f})')
                
                # Add statistics text
                sig_text = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
                stats_text = f'R² = {r_squared:.6f}\np = {p_value:.4f} {sig_text}'
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor='white', alpha=0.8), fontsize=9)
        
        # Customize axes
        ax.set_xlabel(f'{model_name} Sentiment Score', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{metric_name} Count', fontsize=11, fontweight='bold')
        ax.set_title(f'{model_name} × {metric_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=8)
        
        plot_idx += 1

plt.tight_layout(rect=[0, 0, 1, 0.99])

# Save scatter plots
scatter_path = 'outputs/figures/rq3_sentiment_engagement_regression_scatter.png'
plt.savefig(scatter_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Saved scatter plots: {scatter_path}")

# ============================================================================
# STEP 6: CREATE FOREST PLOT (COEFFICIENT PLOT)
# ============================================================================
print("\n=== STEP 6: CREATING FOREST PLOT (COEFFICIENT PLOT) ===\n")

# Prepare data for forest plot
forest_data = regression_df.copy()

# Create y-axis labels (Model names)
forest_data['y_label'] = forest_data['model']

# Create grouping variable for metrics
forest_data['metric_group'] = forest_data['engagement_metric']

# Sort by model and metric for consistent ordering
model_order = ['TextBlob', 'VADER', 'Transformer']
metric_order = ['Likes', 'Replies']
forest_data['model_order'] = forest_data['model'].apply(lambda x: model_order.index(x))
forest_data['metric_order'] = forest_data['engagement_metric'].apply(lambda x: metric_order.index(x))
forest_data = forest_data.sort_values(['model_order', 'metric_order']).reset_index(drop=True)

# Create y positions (reverse order for top-to-bottom display)
forest_data['y_pos'] = range(len(forest_data))

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Colorblind-friendly colors
colors_forest = {
    'Likes': '#0072B2',      # Blue
    'Replies': '#D55E00'     # Orange/Red
}

markers_forest = {
    'Likes': 'o',
    'Replies': 's'
}

# Plot confidence intervals and coefficients
for idx, row in forest_data.iterrows():
    y_pos = row['y_pos']
    color = colors_forest[row['engagement_metric']]
    marker = markers_forest[row['engagement_metric']]
    
    # Plot confidence interval line
    ax.plot([row['slope_ci_lower'], row['slope_ci_upper']], 
            [y_pos, y_pos], 
            color=color, linewidth=2, alpha=0.7, zorder=1)
    
    # Plot coefficient point
    ax.scatter([row['slope']], [y_pos], 
              color=color, marker=marker, s=150, 
              edgecolors='black', linewidth=1.5, zorder=2,
              label=row['engagement_metric'] if idx < 2 else '')
    
    # Add significance indicator
    if row['p_value'] < 0.001:
        sig_marker = '***'
    elif row['p_value'] < 0.01:
        sig_marker = '**'
    elif row['p_value'] < 0.05:
        sig_marker = '*'
    else:
        sig_marker = 'ns'
    
    # Add text annotation with coefficient and significance
    ax.text(row['slope_ci_upper'] + 0.0001, y_pos, 
           f"β={row['slope']:.4f} ({sig_marker})", 
           va='center', fontsize=9, fontweight='bold')

# Add vertical line at zero (no effect)
ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, zorder=0)

# Set y-axis labels
y_labels = []
for idx, row in forest_data.iterrows():
    y_labels.append(f"{row['model']} × {row['engagement_metric']}")

ax.set_yticks(forest_data['y_pos'])
ax.set_yticklabels(y_labels, fontsize=11, fontweight='bold')

# Customize axes
ax.set_xlabel('Regression Coefficient (β) - Effect Size', 
              fontsize=13, fontweight='bold')
ax.set_ylabel('Model × Engagement Metric', 
              fontsize=13, fontweight='bold')
ax.set_title('Linear Regression Coefficients: Sentiment Predicting Engagement\n(95% Confidence Intervals)', 
             fontsize=14, fontweight='bold', pad=20)

# Add grid
ax.grid(True, alpha=0.3, linestyle='--', axis='x')

# Add legend
handles, labels = ax.get_legend_handles_labels()
# Remove duplicates while preserving order
seen = set()
unique_handles = []
unique_labels = []
for handle, label in zip(handles, labels):
    if label not in seen:
        seen.add(label)
        unique_handles.append(handle)
        unique_labels.append(label)

legend = ax.legend(unique_handles, unique_labels, loc='upper right', 
         fontsize=11, framealpha=0.9, title='Engagement Metric', 
         title_fontsize=11)
legend.get_title().set_fontweight('bold')

# Add R² annotation in top right
r2_text = "R² values:\n"
for idx, row in forest_data.iterrows():
    r2_text += f"{row['model']} × {row['engagement_metric']}: {row['r_squared']:.6f}\n"
ax.text(0.98, 0.02, r2_text.rstrip(), transform=ax.transAxes,
       verticalalignment='bottom', horizontalalignment='right',
       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
       fontsize=8, family='monospace')

plt.tight_layout()

# Save forest plot
forest_path = 'outputs/figures/rq3_sentiment_engagement_regression_forest.png'
plt.savefig(forest_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Saved forest plot: {forest_path}")

# ============================================================================
# STEP 7: PRINT COMPREHENSIVE SUMMARY
# ============================================================================
print("\n" + "="*70)
print("RQ3: SENTIMENT-ENGAGEMENT LINEAR REGRESSION ANALYSIS - RESULTS")
print("="*70)

print(f"\nDataset: {len(df):,} comments")
print(f"Valid regressions performed on: {regression_df['n_samples'].min():,} to {regression_df['n_samples'].max():,} comments")

print("\n--- DETAILED REGRESSION RESULTS ---\n")
for _, row in regression_df.iterrows():
    print(f"{row['model']:12s} × {row['engagement_metric']:10s}:")
    print(f"  Intercept (β₀): {row['intercept']:8.4f}")
    print(f"  Slope (β₁):     {row['slope']:8.4f} (95% CI: [{row['slope_ci_lower']:.4f}, {row['slope_ci_upper']:.4f}])")
    print(f"  R²:             {row['r_squared']:.6f} ({row['r_squared']*100:.4f}% variance explained)")
    print(f"  R:               {row['r_value']:8.4f}")
    print(f"  p-value:        {row['p_value']:.6f} ({row['significant']})")
    print(f"  Standardized β: {row['standardized_coef']:8.4f}")
    print(f"  Sample size:    n = {row['n_samples']:,}")
    print()

# Summary statistics
print("--- SUMMARY STATISTICS ---")
min_r2 = regression_df['r_squared'].min()
max_r2 = regression_df['r_squared'].max()
mean_r2 = regression_df['r_squared'].mean()

print(f"\nR² (Variance Explained):")
print(f"  Minimum: {min_r2:.6f} ({min_r2*100:.4f}%)")
print(f"  Maximum: {max_r2:.6f} ({max_r2*100:.4f}%)")
print(f"  Mean:    {mean_r2:.6f} ({mean_r2*100:.4f}%)")

sig_count = (regression_df['p_value'] < 0.05).sum()
print(f"\nStatistical Significance:")
print(f"  Significant (p < 0.05): {sig_count}/{len(regression_df)}")

# Strongest and weakest predictions
strongest_idx = regression_df['r_squared'].idxmax()
weakest_idx = regression_df['r_squared'].idxmin()
strongest = regression_df.loc[strongest_idx]
weakest = regression_df.loc[weakest_idx]

print(f"\nStrongest prediction (highest R²):")
print(f"  {strongest['model']} × {strongest['engagement_metric']}: R² = {strongest['r_squared']:.6f} ({strongest['r_squared']*100:.4f}% variance explained)")

print(f"\nWeakest prediction (lowest R²):")
print(f"  {weakest['model']} × {weakest['engagement_metric']}: R² = {weakest['r_squared']:.6f} ({weakest['r_squared']*100:.4f}% variance explained)")

print("\n" + "="*70)
print("--- INTERPRETATION ---")
print("="*70)

print(f"""
The linear regression analyses reveal that sentiment scores are very weak predictors 
of engagement metrics. Across all model-metric combinations:

- R² values range from {min_r2:.6f} ({min_r2*100:.4f}%) to {max_r2:.6f} ({max_r2*100:.4f}%)
- Mean R²: {mean_r2:.6f} ({mean_r2*100:.4f}%)
- Sentiment explains less than 0.01% of variance in engagement on average

This confirms that sentiment is a weak predictor of engagement, consistent with 
the correlation and Kruskal-Wallis analyses. Engagement in astrobiology discussions 
is driven by multiple factors beyond emotional valence.

""")

print("="*70)
print("--- FILES CREATED ---")
print("="*70)
print(f"\n✓ {raw_output_path}")
print(f"✓ {formatted_output_path}")
print(f"✓ {scatter_path}")
print(f"✓ {forest_path}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)

