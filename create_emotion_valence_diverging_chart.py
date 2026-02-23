#!/usr/bin/env python3
"""
Create Emotion Valence Diverging Chart
======================================

Creates a horizontal diverging bar chart showing emotions grouped by valence
(positive, ambiguous, negative) with the AMBIGUOUS label positioned correctly.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("="*70)
print("CREATING EMOTION VALENCE DIVERGING CHART")
print("="*70)

# ============================================================================
# STEP 1: LOAD EMOTION DATA
# ============================================================================
print("\n=== STEP 1: LOADING EMOTION DATA ===\n")

# Load emotion statistics
emotion_stats_path = 'outputs/emotion_analysis/tables/all_27_emotions_complete_statistics.csv'
if not os.path.exists(emotion_stats_path):
    emotion_stats_path = '../outputs/emotion_analysis/tables/all_27_emotions_complete_statistics.csv'

emotion_df = pd.read_csv(emotion_stats_path)
print(f"✓ Loaded emotion statistics: {len(emotion_df)} emotions")

# ============================================================================
# STEP 2: DEFINE EMOTION VALENCE CATEGORIES
# ============================================================================
print("\n=== STEP 2: CATEGORIZING EMOTIONS BY VALENCE ===\n")

# Define emotion valence based on GoEmotions taxonomy
positive_emotions = [
    'curiosity', 'admiration', 'approval', 'amusement', 'gratitude', 
    'optimism', 'love', 'excitement', 'caring', 'desire', 'joy', 'pride'
]

ambiguous_emotions = [
    'confusion', 'surprise', 'realization'
]

negative_emotions = [
    'annoyance', 'disapproval', 'disappointment', 'fear', 'sadness', 
    'anger', 'remorse', 'disgust', 'nervousness', 'embarrassment', 'grief', 'relief'
]

# Add valence column
def get_valence(emotion):
    if emotion in positive_emotions:
        return 'positive'
    elif emotion in ambiguous_emotions:
        return 'ambiguous'
    elif emotion in negative_emotions:
        return 'negative'
    else:
        return 'unknown'

emotion_df['valence'] = emotion_df['emotion'].apply(get_valence)

# Filter to only include emotions with counts > 0 and known valence
emotion_df = emotion_df[emotion_df['count'] > 0].copy()
emotion_df = emotion_df[emotion_df['valence'] != 'unknown'].copy()

print(f"✓ Categorized emotions:")
print(f"  Positive: {len(emotion_df[emotion_df['valence'] == 'positive'])}")
print(f"  Ambiguous: {len(emotion_df[emotion_df['valence'] == 'ambiguous'])}")
print(f"  Negative: {len(emotion_df[emotion_df['valence'] == 'negative'])}")

# ============================================================================
# STEP 3: PREPARE DATA FOR DIVERGING CHART
# ============================================================================
print("\n=== STEP 3: PREPARING DATA FOR DIVERGING CHART ===\n")

# Sort by valence (positive first, then ambiguous, then negative)
# Within each group, sort by percentage descending
valence_order = {'positive': 0, 'ambiguous': 1, 'negative': 2}
emotion_df['valence_order'] = emotion_df['valence'].map(valence_order)

emotion_df = emotion_df.sort_values(['valence_order', 'percentage'], ascending=[True, False])

# For negative emotions, make percentage negative for diverging effect
emotion_df['percentage_display'] = emotion_df['percentage'].copy()
emotion_df.loc[emotion_df['valence'] == 'negative', 'percentage_display'] = -emotion_df.loc[emotion_df['valence'] == 'negative', 'percentage']

# Calculate totals for each valence
positive_total = emotion_df[emotion_df['valence'] == 'positive']['percentage'].sum()
ambiguous_total = emotion_df[emotion_df['valence'] == 'ambiguous']['percentage'].sum()
negative_total = emotion_df[emotion_df['valence'] == 'negative']['percentage'].sum()

print(f"✓ Prepared data:")
print(f"  Positive total: {positive_total:.1f}%")
print(f"  Ambiguous total: {ambiguous_total:.1f}%")
print(f"  Negative total: {negative_total:.1f}%")

# ============================================================================
# STEP 4: CREATE DIVERGING CHART
# ============================================================================
print("\n=== STEP 4: CREATING DIVERGING CHART ===\n")

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 11

fig, ax = plt.subplots(figsize=(14, 12))

# Get y positions
y_positions = np.arange(len(emotion_df))

# Define colors
positive_color = '#4A90E2'  # Blue
ambiguous_color = '#FF8C42'  # Orange
negative_color = '#E74C3C'   # Red

# Create bars
colors = [positive_color if v == 'positive' else ambiguous_color if v == 'ambiguous' else negative_color 
          for v in emotion_df['valence']]

bars = ax.barh(y_positions, emotion_df['percentage_display'], color=colors, edgecolor='white', linewidth=0.5)

# Add vertical line at 0
ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, zorder=0)

# Set y-axis
ax.set_yticks(y_positions)
ax.set_yticklabels(emotion_df['emotion'], fontsize=10)
ax.invert_yaxis()  # Top emotion at top

# Set x-axis
ax.set_xlabel('Relative Frequency (%)', fontsize=13, fontweight='bold', labelpad=10)
ax.set_xlim(-8, 8)
ax.set_xticks([-7.5, -5, -2.5, 0, 2.5, 5, 7.5])
ax.set_xticklabels(['7.5', '5.0', '2.5', '0.0', '2.5', '5.0', '7.5'])

# Title
ax.set_title('Emotion Valence Distribution in YouTube Astrobiology Comments\n(Based on GoEmotions Taxonomy, Demszky et al., 2020)', 
             fontsize=15, fontweight='bold', pad=20)

# Add percentage labels on bars
for i, (idx, row) in enumerate(emotion_df.iterrows()):
    percentage = row['percentage']
    display_value = row['percentage_display']
    if abs(display_value) > 0.1:  # Only show if > 0.1%
        if display_value > 0:
            ax.text(display_value + 0.15, i, f'{percentage:.2f}%', 
                   va='center', fontsize=9, fontweight='bold')
        else:
            ax.text(display_value - 0.15, i, f'{percentage:.2f}%', 
                   va='center', ha='right', fontsize=9, fontweight='bold')

# Find positions for valence group labels
positive_emotions_df = emotion_df[emotion_df['valence'] == 'positive']
ambiguous_emotions_df = emotion_df[emotion_df['valence'] == 'ambiguous']
negative_emotions_df = emotion_df[emotion_df['valence'] == 'negative']

# Add separator lines between groups (minimal, clean style)
if len(positive_emotions_df) > 0 and len(ambiguous_emotions_df) > 0:
    sep1_y = len(positive_emotions_df) - 0.5
    ax.axhline(y=sep1_y, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)

if len(ambiguous_emotions_df) > 0 and len(negative_emotions_df) > 0:
    sep2_y = len(positive_emotions_df) + len(ambiguous_emotions_df) - 0.5
    ax.axhline(y=sep2_y, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)

# Add valence group labels
# POSITIVE EMOTIONS label on the right
if len(positive_emotions_df) > 0:
    mid_positive = len(positive_emotions_df) / 2 - 0.5
    ax.text(7.0, mid_positive, 'POSITIVE EMOTIONS', 
           fontsize=12, fontweight='bold', color=positive_color,
           ha='right', va='center', rotation=0)

# AMBIGUOUS label - positioned slightly to the left
if len(ambiguous_emotions_df) > 0:
    mid_ambiguous = len(positive_emotions_df) + len(ambiguous_emotions_df) / 2 - 0.5
    # Position slightly to the left (moved from center)
    ax.text(-2.0, mid_ambiguous, 'AMBIGUOUS', 
           fontsize=12, fontweight='bold', color=ambiguous_color,
           ha='left', va='center', rotation=0)

# NEGATIVE EMOTIONS label on the left
if len(negative_emotions_df) > 0:
    mid_negative = len(positive_emotions_df) + len(ambiguous_emotions_df) + len(negative_emotions_df) / 2 - 0.5
    ax.text(-7.0, mid_negative, 'NEGATIVE EMOTIONS', 
           fontsize=12, fontweight='bold', color=negative_color,
           ha='left', va='center', rotation=0)

# Add legend
legend_elements = [
    plt.Rectangle((0,0),1,1, facecolor=positive_color, edgecolor='white', label=f'Positive ({positive_total:.1f}%)'),
    plt.Rectangle((0,0),1,1, facecolor=ambiguous_color, edgecolor='white', label=f'Ambiguous ({ambiguous_total:.1f}%)'),
    plt.Rectangle((0,0),1,1, facecolor=negative_color, edgecolor='white', label=f'Negative ({negative_total:.1f}%)')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.9)

# Add summary box
total_comments = 152501
summary_text = f"Positive: {positive_total:.1f}%\n"
summary_text += f"Ambiguous: {ambiguous_total:.1f}%\n"
summary_text += f"Negative: {negative_total:.1f}%\n"
summary_text += f"n = {total_comments:,} comments"

ax.text(0.98, 0.02, summary_text, transform=ax.transAxes,
        fontsize=10, ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()

# Save
output_path = 'outputs/figures/emotion_diverging_chart.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Saved: {output_path}")
print(f"  AMBIGUOUS label positioned at center (x=0) in the ambiguous emotions section")

print("\n" + "="*70)
print("COMPLETE!")
print("="*70)
