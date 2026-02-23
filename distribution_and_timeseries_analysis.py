"""
Comprehensive Distribution and Time Series Analysis
Addresses supervisor feedback: Show distributions, time series, and identify distribution types
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, normaltest, jarque_bera
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Get project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

print("=" * 80)
print("COMPREHENSIVE DISTRIBUTION AND TIME SERIES ANALYSIS")
print("=" * 80)

# Load data
print("\nLoading data...")
df = pd.read_csv(os.path.join(project_root, 'data/processed/03_sentiment_results.csv'))
print(f"Loaded {len(df):,} comments")

# Convert published_at to datetime
if 'published_at' in df.columns:
    df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
    print("✓ Converted published_at to datetime")

# Ensure numeric columns are numeric
numeric_cols = ['like_count', 'reply_count']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int32')

print("\n" + "=" * 80)
print("PART 1: DISTRIBUTION ANALYSIS")
print("=" * 80)

def analyze_distribution(data, column_name):
    """
    Comprehensive distribution analysis including:
    - Histogram with density curve
    - Q-Q plot for normality assessment
    - Statistical tests (Shapiro-Wilk, D'Agostino, Jarque-Bera)
    - Distribution type identification
    """
    # Convert to numeric and remove NaN
    data_clean = pd.to_numeric(data[column_name], errors='coerce').dropna()
    
    if len(data_clean) == 0:
        return None
    
    # Sample if too large (for normality tests)
    if len(data_clean) > 5000:
        data_sample = data_clean.sample(n=5000, random_state=42)
    else:
        data_sample = data_clean
    
    # Calculate statistics
    mean_val = data_clean.mean()
    median_val = data_clean.median()
    std_val = data_clean.std()
    skewness = data_clean.skew()
    kurtosis = data_clean.kurtosis()
    
    # Count zeros
    zero_count = (data_clean == 0).sum()
    zero_pct = (zero_count / len(data_clean)) * 100
    
    # Statistical tests (on sample if large)
    shapiro_stat, shapiro_p = shapiro(data_sample) if len(data_sample) <= 5000 else (np.nan, np.nan)
    dagostino_stat, dagostino_p = normaltest(data_sample)
    jb_stat, jb_p = jarque_bera(data_sample)
    
    # Identify distribution type
    is_normal = False
    if not np.isnan(shapiro_p) and shapiro_p > 0.05:
        is_normal = True
    
    distribution_type = []
    if zero_pct > 50:
        distribution_type.append("Zero-inflated")
    if abs(skewness) > 1:
        distribution_type.append("Highly skewed")
    elif abs(skewness) > 0.5:
        distribution_type.append("Moderately skewed")
    if skewness > 0:
        distribution_type.append("Right-skewed")
    elif skewness < 0:
        distribution_type.append("Left-skewed")
    if kurtosis > 3:
        distribution_type.append("Heavy-tailed (leptokurtic)")
    elif kurtosis < 3:
        distribution_type.append("Light-tailed (platykurtic)")
    if not is_normal:
        distribution_type.append("Non-normal")
    else:
        distribution_type.append("Approximately normal")
    
    dist_type_str = ", ".join(distribution_type) if distribution_type else "Unknown"
    
    return {
        'column': column_name,
        'n': len(data_clean),
        'mean': mean_val,
        'median': median_val,
        'std': std_val,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'zero_count': zero_count,
        'zero_pct': zero_pct,
        'shapiro_stat': shapiro_stat,
        'shapiro_p': shapiro_p,
        'dagostino_stat': dagostino_stat,
        'dagostino_p': dagostino_p,
        'jb_stat': jb_stat,
        'jb_p': jb_p,
        'is_normal': is_normal,
        'distribution_type': dist_type_str,
        'data': data_clean
    }

# Analyze distributions for engagement metrics
print("\nAnalyzing distributions...")
distribution_results = {}

for col in numeric_cols:
    if col in df.columns:
        print(f"\nAnalyzing {col}...")
        result = analyze_distribution(df, col)
        if result:
            distribution_results[col] = result
            print(f"  Distribution type: {result['distribution_type']}")
            print(f"  Skewness: {result['skewness']:.3f}")
            print(f"  Zero percentage: {result['zero_pct']:.1f}%")
            print(f"  Shapiro-Wilk p-value: {result['shapiro_p']:.6f}")

# Create comprehensive distribution plots
print("\n" + "=" * 80)
print("Creating distribution visualizations...")
print("=" * 80)

for col, result in distribution_results.items():
    data = result['data']
    
    # Create figure with 3 subplots: histogram, density, Q-Q plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Histogram with density curve
    axes[0].hist(data, bins=50, density=True, alpha=0.7, color='steelblue', 
                 edgecolor='black', linewidth=0.5, label='Histogram')
    
    # Add density curve
    if len(data) > 1:
        x_range = np.linspace(data.min(), data.max(), 100)
        try:
            density = stats.gaussian_kde(data)
            axes[0].plot(x_range, density(x_range), 'r-', linewidth=2, label='Density curve')
        except:
            pass
    
    axes[0].axvline(result['mean'], color='green', linestyle='--', linewidth=2, label=f"Mean: {result['mean']:.2f}")
    axes[0].axvline(result['median'], color='orange', linestyle='--', linewidth=2, label=f"Median: {result['median']:.2f}")
    axes[0].set_xlabel(col.replace('_', ' ').title(), fontweight='bold', fontsize=12)
    axes[0].set_ylabel('Density', fontweight='bold', fontsize=12)
    axes[0].set_title(f'{col.replace("_", " ").title()} Distribution\nHistogram with Density Curve', 
                     fontweight='bold', fontsize=13)
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3, linestyle='--')
    
    # Add statistics text box
    stats_text = f"n = {result['n']:,}\n"
    stats_text += f"Mean = {result['mean']:.2f}\n"
    stats_text += f"Median = {result['median']:.2f}\n"
    stats_text += f"Std = {result['std']:.2f}\n"
    stats_text += f"Skewness = {result['skewness']:.3f}\n"
    stats_text += f"Kurtosis = {result['kurtosis']:.3f}\n"
    stats_text += f"Zeros = {result['zero_pct']:.1f}%"
    axes[0].text(0.02, 0.98, stats_text, transform=axes[0].transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                verticalalignment='top', fontsize=9, family='monospace')
    
    # 2. Log-scale histogram (for right-skewed data)
    if result['skewness'] > 0.5:  # Right-skewed
        # Filter zeros for log scale
        data_log = data[data > 0]
        if len(data_log) > 0:
            axes[1].hist(data_log, bins=50, alpha=0.7, color='steelblue', 
                        edgecolor='black', linewidth=0.5)
            axes[1].set_xscale('log')
            axes[1].set_xlabel(col.replace('_', ' ').title() + ' (log scale)', 
                              fontweight='bold', fontsize=12)
            axes[1].set_ylabel('Frequency', fontweight='bold', fontsize=12)
            axes[1].set_title(f'{col.replace("_", " ").title()} Distribution\nLog Scale (excluding zeros)', 
                            fontweight='bold', fontsize=13)
            axes[1].grid(alpha=0.3, linestyle='--')
        else:
            axes[1].text(0.5, 0.5, 'No non-zero values', 
                        transform=axes[1].transAxes, ha='center', va='center')
            axes[1].set_title('Log Scale View', fontweight='bold', fontsize=13)
    else:
        # Regular histogram if not highly skewed
        axes[1].hist(data, bins=50, alpha=0.7, color='steelblue', 
                    edgecolor='black', linewidth=0.5)
        axes[1].set_xlabel(col.replace('_', ' ').title(), fontweight='bold', fontsize=12)
        axes[1].set_ylabel('Frequency', fontweight='bold', fontsize=12)
        axes[1].set_title(f'{col.replace("_", " ").title()} Distribution\nFrequency Histogram', 
                        fontweight='bold', fontsize=13)
        axes[1].grid(alpha=0.3, linestyle='--')
    
    # 3. Q-Q plot for normality assessment
    stats.probplot(data, dist="norm", plot=axes[2])
    axes[2].set_title(f'Q-Q Plot (Normality Test)\n{col.replace("_", " ").title()}', 
                     fontweight='bold', fontsize=13)
    axes[2].grid(alpha=0.3, linestyle='--')
    
    # Add normality test results
    norm_text = f"Shapiro-Wilk:\np = {result['shapiro_p']:.6f}\n\n"
    norm_text += f"D'Agostino:\np = {result['dagostino_p']:.6f}\n\n"
    norm_text += f"Jarque-Bera:\np = {result['jb_p']:.6f}\n\n"
    norm_text += f"Distribution:\n{result['distribution_type']}"
    axes[2].text(0.02, 0.98, norm_text, transform=axes[2].transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                verticalalignment='top', fontsize=8, family='monospace')
    
    plt.tight_layout()
    filename = os.path.join(project_root, f"outputs/figures/comprehensive_distribution_{col}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {filename}")
    plt.close()

# Save distribution analysis summary
dist_summary = pd.DataFrame([
    {
        'variable': col,
        'n': result['n'],
        'mean': result['mean'],
        'median': result['median'],
        'std': result['std'],
        'skewness': result['skewness'],
        'kurtosis': result['kurtosis'],
        'zero_count': result['zero_count'],
        'zero_percentage': result['zero_pct'],
        'shapiro_wilk_p': result['shapiro_p'],
        'dagostino_p': result['dagostino_p'],
        'jarque_bera_p': result['jb_p'],
        'is_normal': result['is_normal'],
        'distribution_type': result['distribution_type']
    }
    for col, result in distribution_results.items()
])

dist_summary.to_csv(os.path.join(project_root, 'outputs/tables/distribution_analysis_summary.csv'), index=False)
print("\n✓ Distribution analysis summary saved")

print("\n" + "=" * 80)
print("PART 2: TIME SERIES ANALYSIS")
print("=" * 80)

if 'published_at' in df.columns and df['published_at'].notna().sum() > 0:
    print("\nCreating time series visualizations...")
    
    # Remove rows with missing dates
    df_time = df[df['published_at'].notna()].copy()
    df_time = df_time.sort_values('published_at')
    
    print(f"Time range: {df_time['published_at'].min()} to {df_time['published_at'].max()}")
    
    # Create time series plots
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # 1. Daily comment counts
    df_time['date'] = df_time['published_at'].dt.date
    daily_counts = df_time.groupby('date').size()
    
    axes[0].plot(daily_counts.index, daily_counts.values, linewidth=1.5, color='steelblue', alpha=0.8)
    axes[0].fill_between(daily_counts.index, daily_counts.values, alpha=0.3, color='steelblue')
    axes[0].set_xlabel('Date', fontweight='bold', fontsize=12)
    axes[0].set_ylabel('Number of Comments', fontweight='bold', fontsize=12)
    axes[0].set_title('Time Series: Daily Comment Count', fontweight='bold', fontsize=14)
    axes[0].grid(alpha=0.3, linestyle='--')
    axes[0].tick_params(axis='x', rotation=45)
    
    # 2. Daily average likes
    daily_likes = df_time.groupby('date')['like_count'].mean()
    axes[1].plot(daily_likes.index, daily_likes.values, linewidth=1.5, color='green', alpha=0.8)
    axes[1].fill_between(daily_likes.index, daily_likes.values, alpha=0.3, color='green')
    axes[1].set_xlabel('Date', fontweight='bold', fontsize=12)
    axes[1].set_ylabel('Average Likes per Comment', fontweight='bold', fontsize=12)
    axes[1].set_title('Time Series: Daily Average Like Count', fontweight='bold', fontsize=14)
    axes[1].grid(alpha=0.3, linestyle='--')
    axes[1].tick_params(axis='x', rotation=45)
    
    # 3. Daily average replies
    daily_replies = df_time.groupby('date')['reply_count'].mean()
    axes[2].plot(daily_replies.index, daily_replies.values, linewidth=1.5, color='orange', alpha=0.8)
    axes[2].fill_between(daily_replies.index, daily_replies.values, alpha=0.3, color='orange')
    axes[2].set_xlabel('Date', fontweight='bold', fontsize=12)
    axes[2].set_ylabel('Average Replies per Comment', fontweight='bold', fontsize=12)
    axes[2].set_title('Time Series: Daily Average Reply Count', fontweight='bold', fontsize=14)
    axes[2].grid(alpha=0.3, linestyle='--')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, 'outputs/figures/timeseries_engagement.png'), dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: timeseries_engagement.png")
    plt.close()
    
    # Time series by sentiment (if sentiment columns exist)
    sentiment_models = ['sentiment_textblob', 'sentiment_vader', 'sentiment_transformer']
    available_models = [m for m in sentiment_models if m in df_time.columns]
    
    if available_models:
        # Use transformer model as primary
        primary_model = 'sentiment_transformer' if 'sentiment_transformer' in df_time.columns else available_models[0]
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # Daily sentiment distribution (proportions)
        daily_sentiment = df_time.groupby(['date', primary_model]).size().unstack(fill_value=0)
        daily_sentiment_pct = daily_sentiment.div(daily_sentiment.sum(axis=1), axis=0) * 100
        
        axes[0].plot(daily_sentiment_pct.index, daily_sentiment_pct.get('positive', 0), 
                    label='Positive', linewidth=1.5, color='green', alpha=0.8)
        axes[0].plot(daily_sentiment_pct.index, daily_sentiment_pct.get('negative', 0), 
                    label='Negative', linewidth=1.5, color='red', alpha=0.8)
        axes[0].plot(daily_sentiment_pct.index, daily_sentiment_pct.get('neutral', 0), 
                    label='Neutral', linewidth=1.5, color='gray', alpha=0.8)
        axes[0].set_xlabel('Date', fontweight='bold', fontsize=12)
        axes[0].set_ylabel('Percentage (%)', fontweight='bold', fontsize=12)
        axes[0].set_title(f'Time Series: Sentiment Distribution Over Time ({primary_model.replace("sentiment_", "").upper()})', 
                         fontweight='bold', fontsize=14)
        axes[0].legend()
        axes[0].grid(alpha=0.3, linestyle='--')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Daily average engagement by sentiment
        daily_engagement_sentiment = df_time.groupby(['date', primary_model])['like_count'].mean().unstack(fill_value=0)
        
        if 'positive' in daily_engagement_sentiment.columns:
            axes[1].plot(daily_engagement_sentiment.index, daily_engagement_sentiment['positive'], 
                        label='Positive', linewidth=1.5, color='green', alpha=0.8)
        if 'negative' in daily_engagement_sentiment.columns:
            axes[1].plot(daily_engagement_sentiment.index, daily_engagement_sentiment['negative'], 
                        label='Negative', linewidth=1.5, color='red', alpha=0.8)
        if 'neutral' in daily_engagement_sentiment.columns:
            axes[1].plot(daily_engagement_sentiment.index, daily_engagement_sentiment['neutral'], 
                        label='Neutral', linewidth=1.5, color='gray', alpha=0.8)
        
        axes[1].set_xlabel('Date', fontweight='bold', fontsize=12)
        axes[1].set_ylabel('Average Likes per Comment', fontweight='bold', fontsize=12)
        axes[1].set_title(f'Time Series: Average Engagement by Sentiment Over Time ({primary_model.replace("sentiment_", "").upper()})', 
                         fontweight='bold', fontsize=14)
        axes[1].legend()
        axes[1].grid(alpha=0.3, linestyle='--')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(project_root, 'outputs/figures/timeseries_sentiment.png'), dpi=300, bbox_inches='tight', facecolor='white')
        print("✓ Saved: timeseries_sentiment.png")
        plt.close()
    
    # Save time series summary statistics
    time_summary = {
        'start_date': df_time['published_at'].min(),
        'end_date': df_time['published_at'].max(),
        'total_days': (df_time['published_at'].max() - df_time['published_at'].min()).days,
        'total_comments': len(df_time),
        'avg_comments_per_day': len(df_time) / max(1, (df_time['published_at'].max() - df_time['published_at'].min()).days)
    }
    
    time_summary_df = pd.DataFrame([time_summary])
    time_summary_df.to_csv(os.path.join(project_root, 'outputs/tables/timeseries_summary.csv'), index=False)
    print("✓ Time series summary saved")
    
else:
    print("WARNING: No valid published_at timestamps found. Skipping time series analysis.")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\nGenerated files:")
print("  - Distribution plots: outputs/figures/comprehensive_distribution_*.png")
print("  - Distribution summary: outputs/tables/distribution_analysis_summary.csv")
if 'published_at' in df.columns and df['published_at'].notna().sum() > 0:
    print("  - Time series plots: outputs/figures/timeseries_*.png")
    print("  - Time series summary: outputs/tables/timeseries_summary.csv")

