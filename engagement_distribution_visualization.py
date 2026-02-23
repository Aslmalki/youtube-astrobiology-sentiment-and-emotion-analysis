"""
Publication-Quality Figure: Engagement Distribution Skewness
=============================================================

Creates a 2x2 multi-panel figure showing the highly skewed, zero-inflated
distribution of engagement metrics (likes and replies) in YouTube comments.

This figure supports Section 3.4 (Data Characteristics) with the statement:
"Our engagement metrics exhibit highly non-normal distributions with three
distinctive characteristics: (1) heavy zero-inflation (62.1% zero likes,
86.1% zero replies), (2) extreme right-skewness (median=0, mean=4.2,
max=12,847 for likes), and (3) heavy tails with extreme outliers."
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import FuncFormatter

# Define formatter function for log-to-linear conversion
def log_to_linear(x, pos):
    """Convert log10 value back to linear scale for display."""
    val = 10**x - 1
    if val < 1:
        return '0'
    elif val < 10:
        return f'{val:.0f}'
    elif val < 1000:
        return f'{val:.0f}'
    else:
        return f'{val:.0f}'

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Create output directory if it doesn't exist
os.makedirs('outputs/figures', exist_ok=True)

print("=" * 70)
print("CREATING ENGAGEMENT DISTRIBUTION VISUALIZATION")
print("=" * 70)
print()

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("=== STEP 1: LOADING DATA ===")
print()

data_file = 'data/processed/03_sentiment_results.csv'
df = pd.read_csv(data_file)

print(f"✓ Loaded {len(df):,} comments")
print(f"  Columns: {len(df.columns)}")
print()

# ============================================================================
# STEP 2: CALCULATE STATISTICS
# ============================================================================
print("=== STEP 2: CALCULATING STATISTICS ===")
print()

# Like count statistics
like_stats = {
    'median': df['like_count'].median(),
    'mean': df['like_count'].mean(),
    'max': df['like_count'].max(),
    'zero_pct': (df['like_count'] == 0).sum() / len(df) * 100,
    'p99': df['like_count'].quantile(0.99),
    'p95': df['like_count'].quantile(0.95),
    'p90': df['like_count'].quantile(0.90)
}

# Reply count statistics
reply_stats = {
    'median': df['reply_count'].median(),
    'mean': df['reply_count'].mean(),
    'max': df['reply_count'].max(),
    'zero_pct': (df['reply_count'] == 0).sum() / len(df) * 100,
    'p99': df['reply_count'].quantile(0.99),
    'p95': df['reply_count'].quantile(0.95),
    'p90': df['reply_count'].quantile(0.90)
}

# Calculate outliers (using 1.5 × IQR rule)
def count_outliers(series, threshold=None):
    """Count outliers using IQR method or custom threshold."""
    if threshold is not None:
        return (series > threshold).sum()
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    return (series > upper_bound).sum()

like_outliers = count_outliers(df['like_count'], threshold=like_stats['p99'])
reply_outliers = count_outliers(df['reply_count'], threshold=reply_stats['p99'])

like_outlier_pct = like_outliers / len(df) * 100
reply_outlier_pct = reply_outliers / len(df) * 100

print("Like Count Statistics:")
print(f"  Median: {like_stats['median']:.1f}")
print(f"  Mean: {like_stats['mean']:.2f}")
print(f"  Max: {like_stats['max']:,}")
print(f"  Zero %: {like_stats['zero_pct']:.1f}%")
print(f"  99th percentile: {like_stats['p99']:.0f}")
print(f"  Outliers (>99th %ile): {like_outliers:,} ({like_outlier_pct:.2f}%)")
print()

print("Reply Count Statistics:")
print(f"  Median: {reply_stats['median']:.1f}")
print(f"  Mean: {reply_stats['mean']:.2f}")
print(f"  Max: {reply_stats['max']:,}")
print(f"  Zero %: {reply_stats['zero_pct']:.1f}%")
print(f"  99th percentile: {reply_stats['p99']:.0f}")
print(f"  Outliers (>99th %ile): {reply_outliers:,} ({reply_outlier_pct:.2f}%)")
print()

# ============================================================================
# STEP 3: CREATE FIGURE
# ============================================================================
print("=== STEP 3: CREATING FIGURE ===")
print()

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Engagement Metrics Distribution: Zero-Inflation and Right-Skewness',
             fontsize=14, fontweight='bold', y=0.995)

# Colors
like_color = '#3498db'  # Blue
reply_color = '#9b59b6'  # Purple

# ============================================================================
# PANEL A: Like Count Histogram
# ============================================================================
ax1 = axes[0, 0]

# Prepare data for log scale (add 1 to handle zeros)
like_log = np.log10(df['like_count'] + 1)

# Create histogram
n_bins = 80
counts, bins, patches = ax1.hist(like_log, bins=n_bins, color=like_color,
                                  alpha=0.7, edgecolor='white', linewidth=0.5)

# Set log scale on x-axis (already log-transformed, so use linear scale)
ax1.set_xlabel('Like Count (log10 scale)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Number of Comments', fontsize=11, fontweight='bold')
ax1.set_title('Like Count Distribution', fontsize=12, fontweight='bold', pad=10)

# Add reference lines
median_log = np.log10(like_stats['median'] + 1)
mean_log = np.log10(like_stats['mean'] + 1)
p99_log = np.log10(like_stats['p99'] + 1)

ax1.axvline(median_log, color='red', linestyle='--', linewidth=2, alpha=0.8,
            label=f"Median = {like_stats['median']:.0f}")
ax1.axvline(mean_log, color='green', linestyle='--', linewidth=2, alpha=0.8,
            label=f"Mean = {like_stats['mean']:.2f}")
ax1.axvline(p99_log, color='orange', linestyle='--', linewidth=2, alpha=0.8,
            label=f"99th %ile = {like_stats['p99']:.0f}")

# Add grid
ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Add text annotations
annotation_text = f"{like_stats['zero_pct']:.1f}% have 0 likes\nMax = {like_stats['max']:,}"
ax1.text(0.98, 0.95, annotation_text,
         transform=ax1.transAxes,
         ha='right', va='top',
         fontsize=9,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=0.5))

# Format x-axis to show original values using FuncFormatter
ax1.xaxis.set_major_formatter(FuncFormatter(log_to_linear))

# ============================================================================
# PANEL B: Like Count Box Plot
# ============================================================================
ax2 = axes[0, 1]

# Prepare data for box plot (log scale)
like_data_log = np.log10(df['like_count'] + 1)

# Create box plot
bp = ax2.boxplot([like_data_log], vert=True, patch_artist=True,
                 widths=0.6, showfliers=True)

# Customize box plot
bp['boxes'][0].set_facecolor(like_color)
bp['boxes'][0].set_alpha(0.7)
bp['boxes'][0].set_edgecolor('black')
bp['boxes'][0].set_linewidth(1.5)

# Customize median line
bp['medians'][0].set_color('red')
bp['medians'][0].set_linewidth(2.5)

# Customize whiskers and caps
for whisker in bp['whiskers']:
    whisker.set_color('black')
    whisker.set_linewidth(1.5)
for cap in bp['caps']:
    cap.set_color('black')
    cap.set_linewidth(1.5)

# Customize outliers
for flier in bp['fliers']:
    flier.set_marker('o')
    flier.set_markerfacecolor('red')
    flier.set_markeredgecolor('red')
    flier.set_alpha(0.3)
    flier.set_markersize(1)

# Set y-axis to log scale (already log-transformed, so use linear)
ax2.set_ylabel('Like Count (log10 scale)', fontsize=11, fontweight='bold')
ax2.set_title('Like Count Box Plot', fontsize=12, fontweight='bold', pad=10)
ax2.set_xticklabels(['Likes'])

# Format y-axis to show original values using FuncFormatter
ax2.yaxis.set_major_formatter(FuncFormatter(log_to_linear))

# Add grid
ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')

# Add text annotation
annotation_text = f"{like_outliers:,} outliers ({like_outlier_pct:.2f}%)\nOutlier threshold: {like_stats['p99']:.0f} likes"
ax2.text(0.98, 0.95, annotation_text,
         transform=ax2.transAxes,
         ha='right', va='top',
         fontsize=9,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=0.5))

# ============================================================================
# PANEL C: Reply Count Histogram
# ============================================================================
ax3 = axes[1, 0]

# Prepare data for log scale (add 1 to handle zeros)
reply_log = np.log10(df['reply_count'] + 1)

# Create histogram
counts, bins, patches = ax3.hist(reply_log, bins=n_bins, color=reply_color,
                                 alpha=0.7, edgecolor='white', linewidth=0.5)

# Set log scale on x-axis
ax3.set_xlabel('Reply Count (log10 scale)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Number of Comments', fontsize=11, fontweight='bold')
ax3.set_title('Reply Count Distribution', fontsize=12, fontweight='bold', pad=10)

# Add reference lines
median_log = np.log10(reply_stats['median'] + 1)
mean_log = np.log10(reply_stats['mean'] + 1)
p99_log = np.log10(reply_stats['p99'] + 1)

ax3.axvline(median_log, color='red', linestyle='--', linewidth=2, alpha=0.8,
            label=f"Median = {reply_stats['median']:.0f}")
ax3.axvline(mean_log, color='green', linestyle='--', linewidth=2, alpha=0.8,
            label=f"Mean = {reply_stats['mean']:.2f}")
ax3.axvline(p99_log, color='orange', linestyle='--', linewidth=2, alpha=0.8,
            label=f"99th %ile = {reply_stats['p99']:.0f}")

# Add grid
ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Add text annotations
annotation_text = f"{reply_stats['zero_pct']:.1f}% have 0 replies\nMax = {reply_stats['max']:,}"
ax3.text(0.98, 0.95, annotation_text,
         transform=ax3.transAxes,
         ha='right', va='top',
         fontsize=9,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=0.5))

# Format x-axis to show original values using FuncFormatter
ax3.xaxis.set_major_formatter(FuncFormatter(log_to_linear))

# ============================================================================
# PANEL D: Reply Count Box Plot
# ============================================================================
ax4 = axes[1, 1]

# Prepare data for box plot (log scale)
reply_data_log = np.log10(df['reply_count'] + 1)

# Create box plot
bp = ax4.boxplot([reply_data_log], vert=True, patch_artist=True,
                 widths=0.6, showfliers=True)

# Customize box plot
bp['boxes'][0].set_facecolor(reply_color)
bp['boxes'][0].set_alpha(0.7)
bp['boxes'][0].set_edgecolor('black')
bp['boxes'][0].set_linewidth(1.5)

# Customize median line
bp['medians'][0].set_color('red')
bp['medians'][0].set_linewidth(2.5)

# Customize whiskers and caps
for whisker in bp['whiskers']:
    whisker.set_color('black')
    whisker.set_linewidth(1.5)
for cap in bp['caps']:
    cap.set_color('black')
    cap.set_linewidth(1.5)

# Customize outliers
for flier in bp['fliers']:
    flier.set_marker('o')
    flier.set_markerfacecolor('red')
    flier.set_markeredgecolor('red')
    flier.set_alpha(0.3)
    flier.set_markersize=1

# Set y-axis
ax4.set_ylabel('Reply Count (log10 scale)', fontsize=11, fontweight='bold')
ax4.set_title('Reply Count Box Plot', fontsize=12, fontweight='bold', pad=10)
ax4.set_xticklabels(['Replies'])

# Format y-axis to show original values using FuncFormatter
ax4.yaxis.set_major_formatter(FuncFormatter(log_to_linear))

# Add grid
ax4.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')

# Add text annotation
annotation_text = f"{reply_outliers:,} outliers ({reply_outlier_pct:.2f}%)\nOutlier threshold: {reply_stats['p99']:.0f} replies"
ax4.text(0.98, 0.95, annotation_text,
         transform=ax4.transAxes,
         ha='right', va='top',
         fontsize=9,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=0.5))

# ============================================================================
# FINALIZE AND SAVE
# ============================================================================

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.98])

# Save figures
png_path = 'outputs/figures/engagement_distribution_skewness.png'
pdf_path = 'outputs/figures/engagement_distribution_skewness.pdf'

plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')

print(f"✓ Saved PNG: {png_path}")
print(f"✓ Saved PDF: {pdf_path}")
print()

print("=" * 70)
print("VISUALIZATION CREATED SUCCESSFULLY")
print("=" * 70)
print()
print("--- FIGURE FEATURES ---")
print("✓ 2x2 multi-panel layout")
print("✓ Log-scale histograms showing zero-inflation and right-skewness")
print("✓ Box plots with outlier visualization")
print("✓ Reference lines (median, mean, 99th percentile)")
print("✓ Statistical annotations")
print("✓ Publication quality (300 DPI)")
print()
print("--- STATISTICS SUMMARY ---")
print(f"Like Count:")
print(f"  Zero-inflation: {like_stats['zero_pct']:.1f}%")
print(f"  Median: {like_stats['median']:.0f}, Mean: {like_stats['mean']:.2f}")
print(f"  Max: {like_stats['max']:,}, 99th %ile: {like_stats['p99']:.0f}")
print()
print(f"Reply Count:")
print(f"  Zero-inflation: {reply_stats['zero_pct']:.1f}%")
print(f"  Median: {reply_stats['median']:.0f}, Mean: {reply_stats['mean']:.2f}")
print(f"  Max: {reply_stats['max']:,}, 99th %ile: {reply_stats['p99']:.0f}")
print()
print("=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)

