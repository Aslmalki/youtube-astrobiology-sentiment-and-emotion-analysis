#!/usr/bin/env python3
"""
RQ3: Enhanced Publication-Ready Forest Plot
Science Communication Audience

Creates an enhanced forest plot with:
- Color coding by significance (gray for non-significant, colored for significant)
- User-friendly axes with directional annotations
- Clean, minimal annotations
- Publication-ready styling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

print("="*70)
print("CREATING ENHANCED PUBLICATION-READY FOREST PLOT")
print("="*70)

# ============================================================================
# STEP 1: LOAD REGRESSION DATA
# ============================================================================
print("\n=== STEP 1: LOADING DATA ===\n")

data_path = 'outputs/tables/rq3_sentiment_engagement_regression.csv'

if not os.path.exists(data_path):
    # Try relative path from notebooks directory
    data_path = '../outputs/tables/rq3_sentiment_engagement_regression.csv'

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Could not find regression data file. Checked: {data_path}")

df = pd.read_csv(data_path)
print(f"✓ Loaded regression data: {len(df)} model-metric combinations")

# ============================================================================
# STEP 2: PREPARE DATA FOR PLOTTING
# ============================================================================
print("\n=== STEP 2: PREPARING DATA ===\n")

# Create significance mask
df['is_significant'] = df['p_value'] < 0.05

# Create label for Y-axis
df['label'] = df.apply(lambda row: f"{row['model']} × {row['engagement_metric']}", axis=1)

# Sort by model and metric for consistent ordering
model_order = ['TextBlob', 'VADER', 'Transformer']
metric_order = ['Likes', 'Replies']
df['model_order'] = df['model'].apply(lambda x: model_order.index(x))
df['metric_order'] = df['engagement_metric'].apply(lambda x: metric_order.index(x))
df = df.sort_values(['model_order', 'metric_order']).reset_index(drop=True)

# Create y positions (reverse order for top-to-bottom display)
df['y_pos'] = range(len(df))

# Colorblind-friendly colors for significant results
colors = {
    'Likes': '#0072B2',      # Blue
    'Replies': '#D55E00'     # Orange/Red
}

# Gray color for non-significant results
gray_color = '#808080'  # Medium gray

print(f"✓ Prepared data for plotting")
print(f"  Significant results: {df['is_significant'].sum()}/{len(df)}")
print(f"  Non-significant results: {(~df['is_significant']).sum()}/{len(df)}")

# ============================================================================
# STEP 3: CREATE ENHANCED FOREST PLOT
# ============================================================================
print("\n=== STEP 3: CREATING ENHANCED FOREST PLOT ===\n")

# Set style
sns.set_theme(style="whitegrid")

# Create figure with appropriate size (wider for better spacing, slightly shorter to reduce bottom space)
fig, ax = plt.subplots(figsize=(14, 8))

# Plot non-significant results first (gray, faded)
non_sig_mask = ~df['is_significant']
if non_sig_mask.any():
    non_sig_data = df[non_sig_mask]
    for idx, row in non_sig_data.iterrows():
        y_pos = row['y_pos']
        # Use errorbar for CI lines and coefficient dot
        ax.errorbar(row['slope'], y_pos,
                   xerr=[[row['slope'] - row['slope_ci_lower']], 
                         [row['slope_ci_upper'] - row['slope']]],
                   fmt='o',  # Circle marker
                   color=gray_color,
                   alpha=0.5,  # Reduced opacity
                   markersize=8,
                   capsize=4,
                   capthick=1.5,
                   elinewidth=1.5,
                   zorder=1,
                   label='_nolegend_')  # Don't add to legend

# Plot significant results (colored, fully opaque, larger)
sig_mask = df['is_significant']
if sig_mask.any():
    sig_data = df[sig_mask]
    for idx, row in sig_data.iterrows():
        y_pos = row['y_pos']
        color = colors[row['engagement_metric']]
        # Use errorbar for CI lines and coefficient dot
        ax.errorbar(row['slope'], y_pos,
                   xerr=[[row['slope'] - row['slope_ci_lower']], 
                         [row['slope_ci_upper'] - row['slope']]],
                   fmt='o',  # Circle marker
                   color=color,
                   alpha=1.0,  # Fully opaque
                   markersize=12,  # Larger for significant
                   capsize=5,
                   capthick=2,
                   elinewidth=2,
                   zorder=2,
                   label=f"Significant ({row['engagement_metric']})" if idx == sig_data.index[0] or row['engagement_metric'] not in [df.loc[i, 'engagement_metric'] for i in sig_data.index[:sig_data.index.get_loc(idx)]] else '_nolegend_')
        
# Add vertical dashed line at x=0 (no effect line)
ax.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.7, zorder=0)

# Set y-axis labels
ax.set_yticks(df['y_pos'])
ax.set_yticklabels(df['label'], fontsize=12, fontweight='bold')

# Customize X-axis
ax.set_xlabel('Association with Engagement (Standardized Beta)',
             fontsize=14, fontweight='bold', labelpad=15)

# Get x-axis limits for annotation placement
# Extend x-axis slightly to accommodate annotations
x_min, x_max = ax.get_xlim()
x_range = x_max - x_min
# Extend x-axis by 15% on the right for annotations
ax.set_xlim(x_min, x_max + x_range * 0.15)

# Update x_max after setting new limits
x_max_new = ax.get_xlim()[1]

# Add coefficient value annotations for significant results (after setting xlim)
# Extend x-axis further to accommodate annotations
x_min_annot, x_max_annot = ax.get_xlim()
x_range_annot = x_max_annot - x_min_annot
ax.set_xlim(x_min_annot, x_max_annot + x_range_annot * 0.25)  # More space for annotations

if sig_mask.any():
    sig_data = df[sig_mask]
    for idx, row in sig_data.iterrows():
        y_pos = row['y_pos']
        color = colors[row['engagement_metric']]
        # Position annotation closer to CI to avoid legend overlap (moved left)
        x_annotation = max(row['slope_ci_upper'], row['slope']) + x_range_annot * 0.05
        # Add coefficient value with appropriate label based on direction
        if row['slope'] > 0:
            annotation_text = f"β = {row['slope']:.3f}\n(Significant Positive Correlation)"
        else:
            annotation_text = f"β = {row['slope']:.3f}\n(Significant Negative Correlation)"
        ax.text(x_annotation, y_pos,
               annotation_text,
               va='center',
               ha='left',
               fontsize=9,
               fontweight='bold',
               color=color,
               linespacing=1.3)

# Label the zero line at the top (with more vertical spacing)
ax.text(0, len(df) - 0.15, 'No Effect Line',
       ha='center',
       va='bottom',
       fontsize=10,
       fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', alpha=0.95, linewidth=1.5),
       zorder=3)

# Add directional annotations below X-axis
# Get updated x limits after extension
x_min_final, x_max_final = ax.get_xlim()
x_range_final = x_max_final - x_min_final

# Negative side annotation (closer to axis)
ax.text(x_min_final + x_range_final * 0.15, -0.5,
       '← Negatively Associated',
       ha='left',
       va='top',
       fontsize=11,
       fontstyle='italic',
       color='#666666',
       transform=ax.get_xaxis_transform())

# Positive side annotation (closer to axis)
ax.text(x_max_final - x_range_final * 0.15, -0.5,
       'Positively Associated →',
       ha='right',
       va='top',
       fontsize=11,
       fontstyle='italic',
       color='#666666',
       transform=ax.get_xaxis_transform())

# Set main title with more spacing
title = "Model Sensitivity Comparison: Transformer Detects Signals Traditional Lexicons Miss"
ax.set_title(title,
            fontsize=15, fontweight='bold', pad=35, y=1.08)

# Add subtitle with proper spacing (further down, no box)
subtitle = "Transformer shows a significant positive association with Likes (p < 0.05), while VADER and TextBlob show no significant link."
ax.text(0.5, 1.02, subtitle,
        transform=ax.transAxes,
        ha='center',
        va='bottom',
        fontsize=10,
        style='italic',
        color='#555555')

# Customize grid (subtle)
ax.grid(True, alpha=0.2, linestyle='--', axis='x', linewidth=0.8)
ax.grid(False, axis='y')  # No vertical grid

# Add color explanation legend in top-right corner
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#0072B2', edgecolor='#0072B2', label='Likes (significant)'),
    Patch(facecolor='#D55E00', edgecolor='#D55E00', label='Replies (significant)'),
    Patch(facecolor='#808080', edgecolor='#808080', alpha=0.5, label='Non-significant')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.95, edgecolor='black', frameon=True)

# Add context footnote about R² (moved down to avoid X-axis labels, no box)
footnote_text = "Note: Overall predictive power (R²) is low (<0.01%) for all models."
ax.text(0.02, -0.12, footnote_text,
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=9,
        style='italic',
        color='#666666')

# Adjust layout to accommodate all elements with reduced bottom spacing
# Leave space at top for title/subtitle, space at bottom for footnote
plt.tight_layout(rect=[0, 0.05, 1, 0.90])

# Create output directory if needed
os.makedirs('outputs/figures', exist_ok=True)

# Save enhanced forest plot
enhanced_forest_path = 'outputs/figures/rq3_sentiment_engagement_regression_forest_enhanced.png'
plt.savefig(enhanced_forest_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Saved enhanced forest plot: {enhanced_forest_path}")

# ============================================================================
# STEP 4: PRINT SUMMARY
# ============================================================================
print("\n" + "="*70)
print("ENHANCED FOREST PLOT CREATED SUCCESSFULLY")
print("="*70)

print("\n--- PLOT FEATURES ---")
print("✓ Color coding by significance:")
print(f"  - Significant (p < 0.05): {sig_mask.sum()} results (colored, fully opaque)")
print(f"  - Non-significant (p ≥ 0.05): {non_sig_mask.sum()} results (gray, 50% opacity)")
print("✓ User-friendly axes:")
print("  - X-axis: 'Association with Engagement (Standardized Beta)'")
print("  - Directional annotations: '← Negatively Associated' and 'Positively Associated →'")
print("  - Zero line labeled: 'No Effect Line'")
print("✓ Narrative alignment:")
print("  - Title focuses on model sensitivity comparison")
print("  - Subtitle highlights Transformer's detection capability")
print("  - Language emphasizes association rather than impact")
print("  - Context footnote about low R² values")
print("✓ Clean annotations:")
print("  - Coefficient values shown only for significant results")
print("  - No complex p-value annotations")
print("✓ De-cluttered:")
print("  - No legend (relies on direct coloring)")
print("  - Clear, descriptive title")

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)

