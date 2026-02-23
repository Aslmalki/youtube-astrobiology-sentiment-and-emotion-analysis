#!/usr/bin/env python3
"""
Execute Notebook 2: Exploratory Data Analysis
Extracted code from notebook cells
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'notebooks'))

# Change to notebooks directory for relative paths
os.chdir(os.path.join(os.path.dirname(__file__), 'notebooks'))

# Cell 1: Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats

np.random.seed(42)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
warnings.filterwarnings('ignore')

try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('seaborn')
sns.set_palette("husl")

print("Libraries imported successfully")

# Load cleaned English dataset
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

df = pd.read_csv('../data/processed/01_comments_english.csv')
print(f"Loaded dataset: {len(df):,} comments, {len(df.columns)} columns")

# Create text features with proper type conversion
print("\n" + "=" * 60)
print("CREATING TEXT FEATURES")
print("=" * 60)

df['text_length'] = df['comment_text_original'].str.len().fillna(0).astype('int32')
df['word_count'] = df['comment_text_original'].str.split().str.len().fillna(0).astype('int32')

print(f"✓ Created text_length (character count)")
print(f"✓ Created word_count (word count)")

# CRITICAL: Convert all numeric columns to proper numeric types
numeric_cols = ['like_count', 'reply_count', 'text_length', 'word_count']
if 'video_view_count' in df.columns:
    numeric_cols.append('video_view_count')

print("\nConverting numeric columns to proper types...")
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    if col != 'video_view_count':  # Keep video_view_count as int64
        df[col] = df[col].astype('int32')
    print(f"✓ Converted {col} to {'int64' if col == 'video_view_count' else 'int32'}")

print(f"\nDataset shape: {df.shape}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Distribution analysis
def plot_distribution_publication_quality(data, column, title, xlabel):
    """
    Create publication-quality distribution plots WITHOUT log scales.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Convert to numeric and filter out NaN
    plot_data = pd.to_numeric(data[column], errors='coerce').dropna()
    
    if len(plot_data) == 0:
        print(f"Warning: No numeric data to plot for {column}")
        return
    
    # Calculate percentiles for zoomed view
    p95 = plot_data.quantile(0.95)
    
    # LEFT: Full distribution (normal scale, all data)
    axes[0].hist(plot_data, bins=50, edgecolor='black', linewidth=0.5,
                 color='steelblue', alpha=0.8)
    axes[0].set_xlabel(xlabel, fontweight='bold', fontsize=12)
    axes[0].set_ylabel('Frequency', fontweight='bold', fontsize=12)
    axes[0].set_title(f'{title} - Full Distribution', fontweight='bold', fontsize=13)
    axes[0].grid(alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add summary text box
    summary_text = f'Mean: {plot_data.mean():.1f}\nMedian: {plot_data.median():.1f}\n95th %ile: {p95:.1f}'
    axes[0].text(0.7, 0.9, summary_text, transform=axes[0].transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                verticalalignment='top', fontsize=9)
    
    # RIGHT: Zoomed to main range (95th percentile)
    main_range_data = plot_data[plot_data <= p95]
    axes[1].hist(main_range_data, bins=50, edgecolor='black', linewidth=0.5,
                 color='steelblue', alpha=0.8)
    axes[1].set_xlabel(xlabel, fontweight='bold', fontsize=12)
    axes[1].set_ylabel('Frequency', fontweight='bold', fontsize=12)
    axes[1].set_title(f'{title} - Main Range (0-95th %ile)', fontweight='bold', fontsize=13)
    axes[1].grid(alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add info text
    pct_in_range = len(main_range_data) / len(plot_data) * 100
    info_text = f'{pct_in_range:.1f}% of data\nRange: 0-{p95:.0f}'
    axes[1].text(0.7, 0.9, info_text, transform=axes[1].transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                verticalalignment='top', fontsize=9)
    
    plt.tight_layout()
    filename = f"../outputs/figures/distribution_{column.lower()}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {filename}")
    plt.close()

# Apply to all numerical variables
print("=" * 60)
print("Distribution Analysis")
print("=" * 60)

plot_distribution_publication_quality(df, 'like_count', 'Like Count Distribution', 'Number of Likes')
plot_distribution_publication_quality(df, 'reply_count', 'Reply Count Distribution', 'Number of Replies')
plot_distribution_publication_quality(df, 'text_length', 'Text Length Distribution', 'Character Count')
plot_distribution_publication_quality(df, 'word_count', 'Word Count Distribution', 'Number of Words')

if 'video_view_count' in df.columns:
    plot_distribution_publication_quality(df, 'video_view_count', 'Video View Count Distribution', 'Number of Views')

# Outlier detection
def detect_outliers_tukey(data, column):
    """
    Detect outliers using Tukey's method (Q1 - 1.5*IQR, Q3 + 1.5*IQR).
    
    CRITICAL FIX: Properly converts data to numeric BEFORE any calculations.
    """
    # CRITICAL: Convert to numeric first and drop NaN
    numeric_data = pd.to_numeric(data[column], errors='coerce').dropna()
    
    if len(numeric_data) == 0:
        return {
            'column': column,
            'count': 0,
            'outliers_count': 0,
            'outliers_pct': 0.0,
            'Q1': 0,
            'Q3': 0,
            'IQR': 0,
            'lower_bound': 0,
            'upper_bound': 0
        }
    
    # Calculate quartiles and IQR
    Q1 = numeric_data.quantile(0.25)
    Q3 = numeric_data.quantile(0.75)
    IQR = Q3 - Q1
    
    # Calculate outlier boundaries
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify outliers
    outliers = (numeric_data < lower_bound) | (numeric_data > upper_bound)
    outliers_count = outliers.sum()
    outliers_pct = (outliers_count / len(numeric_data)) * 100
    
    return {
        'column': column,
        'count': len(numeric_data),
        'outliers_count': outliers_count,
        'outliers_pct': outliers_pct,
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }

# Apply outlier detection to all numerical columns
print("=" * 60)
print("Outlier Detection (Tukey's Method)")
print("=" * 60)

outlier_results = []
for col in numeric_cols:
    result = detect_outliers_tukey(df, col)
    outlier_results.append(result)
    print(f"\n{col}:")
    print(f"  Total values: {result['count']:,}")
    print(f"  Outliers: {result['outliers_count']:,} ({result['outliers_pct']:.2f}%)")
    print(f"  Boundaries: [{result['lower_bound']:.2f}, {result['upper_bound']:.2f}]")

# Save outlier analysis
os.makedirs('../outputs/tables', exist_ok=True)
outlier_df = pd.DataFrame(outlier_results)
outlier_df.to_csv('../outputs/tables/outlier_analysis.csv', index=False)
print("\n✓ Outlier analysis saved to outputs/tables/outlier_analysis.csv")

# Boxplot visualization
print("=" * 60)
print("Boxplot Visualization")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

plot_cols = ['like_count', 'reply_count', 'text_length', 'word_count']

for idx, col in enumerate(plot_cols):
    # Convert to numeric
    plot_data = pd.to_numeric(df[col], errors='coerce').dropna()
    
    if len(plot_data) == 0:
        continue
    
    # Create boxplot
    axes[idx].boxplot([plot_data], vert=True, patch_artist=True,
                     boxprops=dict(facecolor='steelblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     flierprops=dict(marker='o', markerfacecolor='red', 
                                    markersize=3, alpha=0.5))
    
    axes[idx].set_ylabel(col.replace('_', ' ').title(), fontweight='bold', fontsize=12)
    axes[idx].set_title(f'{col.replace("_", " ").title()}', 
                       fontweight='bold', fontsize=13)
    axes[idx].grid(alpha=0.3, linestyle='--', axis='y')
    
    # Use log scale BUT with readable labels (not 10^x)
    axes[idx].set_yscale('log')
    
    # CRITICAL FIX: Set readable tick labels instead of 10^x notation
    y_min = max(plot_data.min(), 1)  # Avoid log(0)
    y_max = plot_data.max()
    
    # Create tick positions at powers of 10
    tick_positions = []
    power = 0
    while 10**power < y_max:
        if 10**power >= y_min:
            tick_positions.append(10**power)
        power += 1
    
    if len(tick_positions) > 0:
        axes[idx].set_yticks(tick_positions)
        # Format labels as regular numbers with commas
        tick_labels = [f'{int(val):,}' if val < 1000 else f'{int(val/1000):,}K' if val < 1000000 else f'{int(val/1000000):,}M' 
                      for val in tick_positions]
        axes[idx].set_yticklabels(tick_labels)
    
    # Add note about log scale
    axes[idx].text(0.5, 0.02, 'Log scale', transform=axes[idx].transAxes,
                  fontsize=8, style='italic', ha='center',
                  bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.savefig('../outputs/figures/outlier_boxplots.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Boxplots saved to outputs/figures/outlier_boxplots.png")
plt.close()

# Correlation analysis
numerical_cols_for_corr = ['like_count', 'reply_count', 'text_length', 'word_count']
if 'video_view_count' in df.columns:
    numerical_cols_for_corr.append('video_view_count')

# CRITICAL: Ensure all columns are numeric
corr_data = df[numerical_cols_for_corr].copy()
for col in numerical_cols_for_corr:
    corr_data[col] = pd.to_numeric(corr_data[col], errors='coerce')

# Drop rows with any NaN values
corr_data = corr_data.dropna()

# Calculate correlation matrices
corr_spearman = corr_data.corr(method='spearman')
corr_pearson = corr_data.corr(method='pearson')

# Create heatmap visualization
print("=" * 60)
print("Correlation Analysis")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Spearman correlation
sns.heatmap(corr_spearman, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=axes[0],
            vmin=-1, vmax=1)
axes[0].set_title('Spearman Correlation Matrix\n(For Skewed Distributions)', 
                 fontweight='bold', fontsize=14, pad=15)

# Pearson correlation
sns.heatmap(corr_pearson, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=axes[1],
            vmin=-1, vmax=1)
axes[1].set_title('Pearson Correlation Matrix\n(Linear Relationships)', 
                 fontweight='bold', fontsize=14, pad=15)

plt.tight_layout()
plt.savefig('../outputs/figures/correlation_matrix.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Correlation matrices saved")
plt.close()

# Save correlation tables
corr_spearman.to_csv('../outputs/tables/correlation_spearman.csv')
corr_pearson.to_csv('../outputs/tables/correlation_pearson.csv')
print("✓ Correlation tables saved")

# Topic distribution analysis
print("=" * 60)
print("Topic Distribution Analysis")
print("=" * 60)

if 'search_query' in df.columns:
    topic_counts = df['search_query'].value_counts()
    topic_percentages = (topic_counts / len(df) * 100).round(2)
    
    topic_distribution = pd.DataFrame({
        'topic': topic_counts.index,
        'count': topic_counts.values,
        'percentage': topic_percentages.values
    })
    
    print(topic_distribution.to_string(index=False))
    
    # Save table
    topic_distribution.to_csv('../outputs/tables/topic_distribution.csv', index=False)
    print("\n✓ Topic distribution table saved")
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(topic_distribution['topic'], topic_distribution['count'],
                   color='steelblue', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Number of Comments', fontweight='bold', fontsize=12)
    ax.set_ylabel('Topic', fontweight='bold', fontsize=12)
    ax.set_title('Comment Distribution Across Astrobiology Topics', 
                fontweight='bold', fontsize=14, pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (idx, row) in enumerate(topic_distribution.iterrows()):
        ax.text(row['count'] + max(topic_distribution['count']) * 0.01, i,
                f"{int(row['count']):,} ({row['percentage']:.1f}%)",
                va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('../outputs/figures/topic_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Topic distribution saved")
    plt.close()
else:
    print("WARNING: 'search_query' column not found in dataset")

# Summary statistics
def calculate_summary_stats(data, column):
    """
    Calculate appropriate summary statistics based on distribution shape.
    
    CRITICAL: Converts data to numeric first.
    """
    # Convert to numeric and drop NaN
    numeric_data = pd.to_numeric(data[column], errors='coerce').dropna()
    
    if len(numeric_data) == 0:
        return None
    
    stats_dict = {
        'column': column,
        'count': len(numeric_data),
        'mean': numeric_data.mean(),
        'median': numeric_data.median(),
        'std': numeric_data.std(),
        'min': numeric_data.min(),
        'max': numeric_data.max(),
        'Q1': numeric_data.quantile(0.25),
        'Q3': numeric_data.quantile(0.75),
        'IQR': numeric_data.quantile(0.75) - numeric_data.quantile(0.25),
        'skewness': numeric_data.skew(),
        'kurtosis': numeric_data.kurtosis()
    }
    
    # Determine if distribution is symmetrical
    is_symmetrical = abs(stats_dict['skewness']) < 0.5 and stats_dict['count'] > 30
    
    if is_symmetrical:
        stats_dict['summary'] = f"{stats_dict['mean']:.2f} ± {stats_dict['std']:.2f} (mean ± SD)"
        stats_dict['stat_type'] = 'mean_sd'
    else:
        stats_dict['summary'] = f"{stats_dict['median']:.2f} [{stats_dict['Q1']:.2f}, {stats_dict['Q3']:.2f}] (median [IQR])"
        stats_dict['stat_type'] = 'median_iqr'
    
    return stats_dict

# Calculate for all numerical columns
print("=" * 60)
print("Summary Statistics")
print("=" * 60)

summary_results = []
for col in numeric_cols:
    result = calculate_summary_stats(df, col)
    if result:
        summary_results.append(result)
        print(f"\n{col}:")
        print(f"  {result['summary']}")
        print(f"  Skewness: {result['skewness']:.3f}, Kurtosis: {result['kurtosis']:.3f}")
        print(f"  Range: [{result['min']:.2f}, {result['max']:.2f}]")

summary_df = pd.DataFrame(summary_results)
summary_df.to_csv('../outputs/tables/summary_statistics.csv', index=False)
print("\n✓ Summary statistics saved")
print("\n" + "=" * 60)
print("NOTEBOOK 2 COMPLETE")
print("=" * 60)
print("Next step: Run Notebook 3 (Preprocessing and Feature Engineering)")

