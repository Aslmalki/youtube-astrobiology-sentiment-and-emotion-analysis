#!/usr/bin/env python3
"""
Create Supplementary Figure and Table for All 27 Emotions
==========================================================

This script creates:
1. A supplementary figure showing all 27 emotions with percentage frequencies
2. A CSV table listing all emotions with counts and percentages
3. A relative frequencies (%) chart for easy comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("="*70)
print("CREATING ALL 27 EMOTIONS FIGURE AND TABLE")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================
EMOTION_MODEL = "SamLowe/roberta-base-go_emotions"
output_dir = 'outputs/emotion_analysis'
os.makedirs(f'{output_dir}/figures', exist_ok=True)
os.makedirs(f'{output_dir}/tables', exist_ok=True)

# ============================================================================
# STEP 1: GET ALL 27 EMOTION LABELS FROM MODEL
# ============================================================================
print("\n=== STEP 1: GETTING ALL EMOTION LABELS ===\n")

# Load model config to get all emotion labels
print(f"Loading model config: {EMOTION_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL)

# Get all emotion labels (should be 27 + neutral = 28 total)
emotion_labels = model.config.id2label
all_emotion_names = [emotion_labels[i] for i in sorted(emotion_labels.keys())]

print(f"✓ Found {len(all_emotion_names)} emotion categories")
print(f"  Emotions: {', '.join(all_emotion_names)}")

# ============================================================================
# STEP 2: LOAD EMOTION DATA
# ============================================================================
print("\n=== STEP 2: LOADING EMOTION DATA ===\n")

emotion_data_path = 'data/processed/youtube_comments_with_emotions.csv'
if not os.path.exists(emotion_data_path):
    emotion_data_path = '../data/processed/youtube_comments_with_emotions.csv'

if not os.path.exists(emotion_data_path):
    raise FileNotFoundError(f"Emotion data file not found: {emotion_data_path}")

print(f"Loading emotion data from: {emotion_data_path}")
df_emotions = pd.read_csv(emotion_data_path)

print(f"✓ Loaded {len(df_emotions):,} comments with emotion data")

# ============================================================================
# STEP 3: CALCULATE COUNTS AND PERCENTAGES FOR ALL EMOTIONS
# ============================================================================
print("\n=== STEP 3: CALCULATING EMOTION STATISTICS ===\n")

# Get dominant emotion counts
dominant_emotion_counts = df_emotions['dominant_emotion'].value_counts()

# Separate neutral from the 27 emotions
total_comments = len(df_emotions)
emotion_stats = []
neutral_count = 0
neutral_percentage = 0.0

for emotion in all_emotion_names:
    count = dominant_emotion_counts.get(emotion, 0)
    percentage = (count / total_comments * 100) if total_comments > 0 else 0.0
    
    if emotion == 'neutral':
        neutral_count = count
        neutral_percentage = round(percentage, 2)
    else:
        # Only include the 27 non-neutral emotions
        emotion_stats.append({
            'emotion': emotion,
            'count': count,
            'percentage': round(percentage, 2)
        })

# Create DataFrame and sort by count (descending)
emotion_df = pd.DataFrame(emotion_stats)
emotion_df = emotion_df.sort_values('count', ascending=False).reset_index(drop=True)

# Add a note about neutral in the summary
print(f"\nNote: 'neutral' category has {neutral_count:,} comments ({neutral_percentage}%)")
print(f"  This is excluded from the 27 emotions analysis.")

print(f"✓ Calculated statistics for {len(emotion_df)} emotions")
print(f"\nTop 5 emotions:")
print(emotion_df.head(5).to_string(index=False))
print(f"\nBottom 5 emotions:")
print(emotion_df.tail(5).to_string(index=False))

# ============================================================================
# STEP 4: SAVE CSV TABLE
# ============================================================================
print("\n=== STEP 4: SAVING CSV TABLE ===\n")

csv_output_path = f'{output_dir}/tables/all_27_emotions_complete_statistics.csv'
emotion_df.to_csv(csv_output_path, index=False)

print(f"✓ Saved CSV table to: {csv_output_path}")
print(f"  Total emotions: {len(emotion_df)}")
print(f"  Total comments: {total_comments:,}")

# ============================================================================
# STEP 5: CREATE VISUALIZATION - ALL 27 EMOTIONS WITH PERCENTAGES
# ============================================================================
print("\n=== STEP 5: CREATING VISUALIZATION ===\n")

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

# Create figure with appropriate size for 27 emotions
fig, ax = plt.subplots(figsize=(10, 14))

# Create horizontal bar chart
colors = plt.cm.viridis(np.linspace(0, 1, len(emotion_df)))
bars = ax.barh(range(len(emotion_df)), emotion_df['percentage'], color=colors)

# Customize y-axis
ax.set_yticks(range(len(emotion_df)))
ax.set_yticklabels(emotion_df['emotion'], fontsize=9)
ax.invert_yaxis()  # Top emotion at top

# Add percentage labels on bars
for i, (idx, row) in enumerate(emotion_df.iterrows()):
    percentage = row['percentage']
    count = row['count']
    # Only show label if percentage > 0.1% to avoid clutter
    if percentage > 0.1:
        ax.text(percentage + 0.05, i, f'{percentage:.2f}%', 
               va='center', fontsize=8, fontweight='bold')
    else:
        # For very small percentages, show count instead
        ax.text(0.1, i, f'{count} ({percentage:.2f}%)', 
               va='center', fontsize=7, style='italic')

# Labels and title
ax.set_xlabel('Relative Frequency (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Emotion', fontsize=12, fontweight='bold')
ax.set_title('Distribution of All 27 Emotions in YouTube Comments\n(Percentage Frequencies)', 
             fontsize=14, fontweight='bold', pad=20)

# Add grid for better readability
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Add summary statistics as text
total_emotions_detected = (emotion_df['count'] > 0).sum()
summary_text = f"27 Emotions Detected: {total_emotions_detected} of {len(emotion_df)}\n"
summary_text += f"Total comments: {total_comments:,}\n"
summary_text += f"Note: 'neutral' category ({neutral_percentage}%) excluded"
ax.text(0.98, 0.02, summary_text, transform=ax.transAxes,
        fontsize=9, ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Adjust layout
plt.tight_layout()

# Save figure
figure_output_path = f'{output_dir}/figures/supplementary_all_27_emotions_percentages.png'
plt.savefig(figure_output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved figure to: {figure_output_path}")

# ============================================================================
# STEP 6: CREATE ALTERNATIVE VISUALIZATION - COMPACT COMPARISON CHART
# ============================================================================
print("\n=== STEP 6: CREATING COMPACT COMPARISON CHART ===\n")

# Create a more compact version for easier comparison
fig2, ax2 = plt.subplots(figsize=(12, 8))

# Create horizontal bar chart with sorted data
bars2 = ax2.barh(range(len(emotion_df)), emotion_df['percentage'], 
                 color=plt.cm.plasma(np.linspace(0, 0.8, len(emotion_df))))

# Add value labels
for i, (idx, row) in enumerate(emotion_df.iterrows()):
    percentage = row['percentage']
    if percentage > 0.5:  # Show label for emotions > 0.5%
        ax2.text(percentage + 0.1, i, f'{percentage:.2f}%', 
                va='center', fontsize=8, fontweight='bold')
    elif percentage > 0:
        ax2.text(percentage + 0.05, i, f'{percentage:.2f}%', 
                va='center', fontsize=7)

# Customize
ax2.set_yticks(range(len(emotion_df)))
ax2.set_yticklabels(emotion_df['emotion'], fontsize=9)
ax2.invert_yaxis()
ax2.set_xlabel('Relative Frequency (%)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Emotion', fontsize=12, fontweight='bold')
ax2.set_title('All 27 Emotions: Relative Frequency Comparison\n(Ordered by Frequency, Neutral Excluded)', 
              fontsize=14, fontweight='bold', pad=15)
ax2.grid(axis='x', alpha=0.3, linestyle='--')
ax2.set_axisbelow(True)

# Add vertical line at 1% for reference
max_percentage = emotion_df['percentage'].max()
ax2.axvline(x=1.0, color='red', linestyle=':', alpha=0.5, linewidth=1)
ax2.text(1.0, len(emotion_df) - 1, '1%', rotation=90, va='bottom', 
         ha='right', fontsize=8, color='red', alpha=0.7)

plt.tight_layout()

# Save compact version
compact_output_path = f'{output_dir}/figures/supplementary_all_27_emotions_comparison.png'
plt.savefig(compact_output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved compact comparison chart to: {compact_output_path}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\n✓ CSV Table: {csv_output_path}")
print(f"  - {len(emotion_df)} emotions listed (27 non-neutral emotions)")
print(f"  - Columns: emotion, count, percentage")
print(f"  - Note: 'neutral' category ({neutral_count:,} comments, {neutral_percentage}%) excluded")
print(f"\n✓ Figure 1 (Full): {figure_output_path}")
print(f"  - All 27 emotions with percentage frequencies")
print(f"  - Size: 10x14 inches, 300 DPI")
print(f"\n✓ Figure 2 (Compact): {compact_output_path}")
print(f"  - Compact comparison chart for easy viewing")
print(f"  - Size: 12x8 inches, 300 DPI")
print(f"\nTotal comments analyzed: {total_comments:,}")
print(f"Emotions with counts > 0: {total_emotions_detected} of {len(emotion_df)}")
print("\n" + "="*70)
print("COMPLETE!")
print("="*70)

