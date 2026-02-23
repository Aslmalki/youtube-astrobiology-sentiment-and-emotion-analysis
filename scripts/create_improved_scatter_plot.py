"""
Create Improved Scatter Plot for Outlier Analysis
Publication-ready visualization with all requested improvements
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

# Get project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("=" * 80)
print("CREATING IMPROVED SCATTER PLOT")
print("=" * 80)

# Load data
print("\nLoading data...")
df = pd.read_csv(os.path.join(project_root, 'data/processed/03_sentiment_results.csv'))
print(f"Loaded {len(df):,} comments")

# Ensure numeric columns are numeric
df['like_count'] = pd.to_numeric(df['like_count'], errors='coerce').fillna(0).astype('int32')
df['reply_count'] = pd.to_numeric(df['reply_count'], errors='coerce').fillna(0).astype('int32')

# Define thresholds (99th percentile)
like_threshold = 115
reply_threshold = 14

print(f"\nThresholds:")
print(f"  Likes: {like_threshold}")
print(f"  Replies: {reply_threshold}")

# Classify comments
df['is_outlier'] = (df['like_count'] > like_threshold) | (df['reply_count'] > reply_threshold)

# Separate typical and outliers
typical = df[~df['is_outlier']].copy()
outliers = df[df['is_outlier']].copy()

print(f"\nClassification:")
print(f"  Typical comments: {len(typical):,}")
print(f"  Outliers: {len(outliers):,}")

# Replace zeros with 1 for log scale (log(0) is undefined)
typical_likes = typical['like_count'].replace(0, 1)
typical_replies = typical['reply_count'].replace(0, 1)
outliers_likes = outliers['like_count'].replace(0, 1)
outliers_replies = outliers['reply_count'].replace(0, 1)

def create_scatter_plot(with_shade=True):
    """Create improved scatter plot with all enhancements"""
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    
    # Add shaded region for typical zone (if requested)
    if with_shade:
        # For log scale, create a rectangle using fill_between
        # This creates a rectangle from (1,1) to (like_threshold, reply_threshold)
        x_shade = np.logspace(0, np.log10(like_threshold), 100)
        y_shade_low = np.ones_like(x_shade)
        y_shade_high = np.full_like(x_shade, reply_threshold)
        ax.fill_between(x_shade, y_shade_low, y_shade_high, 
                       alpha=0.1, color='gray', zorder=0, label='_nolegend_')
    
    # Plot typical comments (behind)
    ax.scatter(typical_likes, typical_replies, 
               s=10, alpha=0.3, color='#ADD8E6', 
               label=f'Typical Comments (n={len(typical):,})', 
               zorder=1, edgecolors='none')
    
    # Plot outliers (on top)
    ax.scatter(outliers_likes, outliers_replies, 
               s=30, alpha=0.7, color='#E74C3C', 
               label=f'Outliers (n={len(outliers):,})', 
               zorder=2, edgecolors='darkred', linewidth=0.5)
    
    # Add reference lines (99th percentile thresholds)
    ax.axvline(x=like_threshold, color='black', linestyle='--', 
               linewidth=2, alpha=0.7, zorder=3, label='_nolegend_')
    ax.axhline(y=reply_threshold, color='black', linestyle='--', 
               linewidth=2, alpha=0.7, zorder=3, label='_nolegend_')
    
    # Add annotation for thresholds
    # Position annotation in upper right area
    ax.text(like_threshold * 1.5, 800, '99th percentile\nthresholds', 
            fontsize=11, ha='left', va='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=1),
            zorder=4)
    
    # Set log scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Set axis limits
    ax.set_xlim(1, 10**5)
    ax.set_ylim(1, 10**3)
    
    # Set axis labels
    ax.set_xlabel('Like Count (log scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reply Count (log scale)', fontsize=12, fontweight='bold')
    
    # Set title
    ax.set_title('Engagement Scatter Plot: Likes vs Replies\n(Outliers Highlighted)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, zorder=0)
    
    # Set tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    
    # Format log scale ticks to be more readable
    from matplotlib.ticker import LogFormatterSciNotation
    ax.xaxis.set_major_formatter(LogFormatterSciNotation())
    ax.yaxis.set_major_formatter(LogFormatterSciNotation())
    
    # Legend
    legend = ax.legend(loc='upper left', fontsize=11, framealpha=0.8, 
                     frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(0.5)
    
    # Tight layout
    plt.tight_layout()
    
    return fig, ax

# Create version WITH shaded region
print("\nCreating scatter plot WITH shaded region...")
fig1, ax1 = create_scatter_plot(with_shade=True)
output_path1 = os.path.join(project_root, 'outlier_analysis', 'scatter_likes_vs_replies_improved.png')
fig1.savefig(output_path1, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig1)
print(f"✓ Saved: {output_path1}")

# Create version WITHOUT shaded region
print("\nCreating scatter plot WITHOUT shaded region...")
fig2, ax2 = create_scatter_plot(with_shade=False)
output_path2 = os.path.join(project_root, 'outlier_analysis', 'scatter_likes_vs_replies_improved_no_shade.png')
fig2.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig2)
print(f"✓ Saved: {output_path2}")

print("\n" + "=" * 80)
print("IMPROVED SCATTER PLOTS CREATED")
print("=" * 80)
print("\nBoth versions saved:")
print(f"  1. With shaded region: {output_path1}")
print(f"  2. Without shaded region: {output_path2}")
print("\nRecommendation: The version WITHOUT the shaded region is cleaner and more")
print("publication-ready. The reference lines and color distinction are sufficient")
print("to show the outlier region clearly.")

# Print some statistics for verification
print("\n" + "=" * 80)
print("DATA VERIFICATION")
print("=" * 80)
print(f"\nTypical comments:")
print(f"  Likes range: {typical['like_count'].min()} to {typical['like_count'].max()}")
print(f"  Replies range: {typical['reply_count'].min()} to {typical['reply_count'].max()}")
print(f"\nOutliers:")
print(f"  Likes range: {outliers['like_count'].min()} to {outliers['like_count'].max()}")
print(f"  Replies range: {outliers['reply_count'].min()} to {outliers['reply_count'].max()}")
print(f"  Mean likes: {outliers['like_count'].mean():.2f}")
print(f"  Mean replies: {outliers['reply_count'].mean():.2f}")

