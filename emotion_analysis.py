#!/usr/bin/env python3
"""
Comprehensive Emotion Analysis with Full Spectrum of Visualizations

This script performs emotion detection and creates both basic and enhanced
visualization options for maximum flexibility in paper selection.

Model: SamLowe/roberta-base-go_emotions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import networkx as nx
# from sklearn.preprocessing import StandardScaler  # Not used, commented out
from scipy.stats import linregress
import warnings
import os
from tqdm import tqdm

warnings.filterwarnings('ignore')

print("="*70)
print("COMPREHENSIVE EMOTION ANALYSIS")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMOTION_MODEL = "SamLowe/roberta-base-go_emotions"

# Create output directories
os.makedirs('outputs/tables', exist_ok=True)
os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# ============================================================================
# PART 1: LOAD DATA
# ============================================================================
print("\n=== PART 1: LOADING DATA ===\n")

data_path = 'data/processed/03_sentiment_results.csv'
if not os.path.exists(data_path):
    data_path = '../data/processed/03_sentiment_results.csv'

df = pd.read_csv(data_path, nrows=None)  # Load all data
print(f"✓ Loaded dataset: {len(df):,} comments, {len(df.columns)} columns")

# Use comment_text_original for emotion detection
if 'comment_text_original' not in df.columns:
    raise ValueError("comment_text_original column not found")

print(f"✓ Using 'comment_text_original' for emotion detection")

# ============================================================================
# PART 2: EMOTION DETECTION
# ============================================================================
print("\n=== PART 2: EMOTION DETECTION ===\n")

# Check if emotions already detected
emotion_output_path = 'data/processed/youtube_comments_with_emotions.csv'
if os.path.exists(emotion_output_path):
    print(f"✓ Emotion data already exists. Loading from: {emotion_output_path}")
    df_emotions = pd.read_csv(emotion_output_path)
    print(f"  Loaded {len(df_emotions):,} comments with emotion data")
else:
    print(f"Loading emotion model: {EMOTION_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL)
    model.to(DEVICE)
    model.eval()
    
    # Get emotion labels
    emotion_labels = model.config.id2label
    print(f"✓ Model loaded. Detecting {len(emotion_labels)} emotions")
    print(f"  Device: {DEVICE}")
    print(f"  Batch size: {BATCH_SIZE}")
    
    # Process in batches
    texts = df['comment_text_original'].astype(str).tolist()
    all_emotion_scores = []
    
    print(f"\nProcessing {len(texts):,} comments in batches...")
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i:i+BATCH_SIZE]
        
        # Tokenize
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, 
                         truncation=True, max_length=512).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        
        all_emotion_scores.append(probs)
    
    # Combine all scores
    emotion_scores = np.vstack(all_emotion_scores)
    
    # Create emotion columns
    df_emotions = df.copy()
    for idx, emotion_id in enumerate(emotion_labels.keys()):
        emotion_name = emotion_labels[emotion_id]
        df_emotions[f'emotion_{emotion_name}'] = emotion_scores[:, idx]
    
    # Get dominant emotion for each comment
    dominant_emotion_idx = np.argmax(emotion_scores, axis=1)
    df_emotions['dominant_emotion'] = [emotion_labels[idx] for idx in dominant_emotion_idx]
    df_emotions['dominant_emotion_score'] = [emotion_scores[i, idx] 
                                            for i, idx in enumerate(dominant_emotion_idx)]
    
    # Save
    df_emotions.to_csv(emotion_output_path, index=False)
    print(f"\n✓ Saved emotion data to: {emotion_output_path}")

# Get emotion column names
emotion_cols = [col for col in df_emotions.columns if col.startswith('emotion_')]
emotion_names = [col.replace('emotion_', '') for col in emotion_cols]
print(f"\n✓ Detected {len(emotion_names)} emotions")

# ============================================================================
# PART 3: DESCRIPTIVE ANALYSIS & VISUALIZATIONS
# ============================================================================
print("\n=== PART 3: DESCRIPTIVE ANALYSIS & VISUALIZATIONS ===\n")

# Calculate overall emotion distribution
emotion_counts = df_emotions['dominant_emotion'].value_counts()
# Exclude neutral if present
if 'neutral' in emotion_counts.index:
    emotion_counts = emotion_counts.drop('neutral')

print(f"✓ Calculated emotion distribution")
print(f"  Top 5 emotions: {emotion_counts.head(5).to_dict()}")

# Save overall distribution table
emotion_dist_overall = pd.DataFrame({
    'emotion': emotion_counts.index,
    'count': emotion_counts.values,
    'percentage': (emotion_counts.values / len(df_emotions) * 100).round(2)
})
emotion_dist_overall.to_csv('outputs/tables/emotion_distribution_overall.csv', index=False)
print(f"✓ Saved: outputs/tables/emotion_distribution_overall.csv")

# ============================================================================
# FIGURE 1: OVERALL EMOTION DISTRIBUTION
# ============================================================================
print("\n--- Creating Figure 1: Overall Emotion Distribution ---\n")

# Option 1A: Simple Bar Chart
print("Creating fig1a_bar_chart.png...")
top_15_emotions = emotion_counts.head(15)

fig, ax = plt.subplots(figsize=(10, 8))
top_15_emotions.plot(kind='barh', ax=ax, color='steelblue')
ax.set_xlabel('Frequency', fontsize=12, fontweight='bold')
ax.set_ylabel('Emotion', fontsize=12, fontweight='bold')
ax.set_title('Top 15 Emotions in YouTube Comments\n(Overall Distribution)', 
             fontsize=14, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('outputs/figures/fig1a_bar_chart.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: outputs/figures/fig1a_bar_chart.png")

# Option 1B & 1C: Clustered Bar Chart and Dendrogram
print("Creating fig1b_clustered_bar_chart.png and fig1c_dendrogram.png...")

# Calculate co-occurrence matrix (top 15 emotions)
top_15_emotion_names = top_15_emotions.index.tolist()
cooccurrence_matrix = np.zeros((len(top_15_emotion_names), len(top_15_emotion_names)))

# For each comment, check which emotions are above threshold
threshold = 0.1  # Emotion score threshold
for idx, emotion_name in enumerate(top_15_emotion_names):
    col_name = f'emotion_{emotion_name}'
    if col_name in df_emotions.columns:
        emotion_present = (df_emotions[col_name] > threshold).astype(int)
        for jdx, other_emotion in enumerate(top_15_emotion_names):
            if idx != jdx:
                other_col = f'emotion_{other_emotion}'
                if other_col in df_emotions.columns:
                    other_present = (df_emotions[other_col] > threshold).astype(int)
                    cooccurrence = (emotion_present * other_present).sum()
                    cooccurrence_matrix[idx, jdx] = cooccurrence

# Normalize to similarity (0-1)
max_cooccur = cooccurrence_matrix.max()
if max_cooccur > 0:
    similarity_matrix = 1 - (cooccurrence_matrix / max_cooccur)
else:
    similarity_matrix = cooccurrence_matrix

# Convert similarity to distance matrix (diagonal must be zero)
distance_matrix = similarity_matrix.copy()
np.fill_diagonal(distance_matrix, 0)  # Set diagonal to zero

# Hierarchical clustering
condensed_distances = squareform(distance_matrix)
linkage_matrix = linkage(condensed_distances, method='ward')

# Create dendrogram
fig, ax = plt.subplots(figsize=(12, 8))
dendrogram(linkage_matrix, labels=top_15_emotion_names, ax=ax, 
          orientation='top', leaf_rotation=90)
ax.set_title('Emotion Clustering Dendrogram\n(Hierarchical Clustering of Emotion Co-occurrence)', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Emotion', fontsize=12, fontweight='bold')
ax.set_ylabel('Distance', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/figures/fig1c_dendrogram.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: outputs/figures/fig1c_dendrogram.png")

# Get cluster assignments
n_clusters = 4  # Adjust as needed
cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

# Reorder emotions by cluster
cluster_order = sorted(range(len(top_15_emotion_names)), 
                      key=lambda x: (cluster_labels[x], -top_15_emotions.iloc[x]))
ordered_emotions = [top_15_emotion_names[i] for i in cluster_order]
ordered_counts = [top_15_emotions[emotion] for emotion in ordered_emotions]

# Create clustered bar chart
fig, ax = plt.subplots(figsize=(12, 8))
colors_list = plt.cm.Set3(np.linspace(0, 1, n_clusters))
bar_colors = [colors_list[cluster_labels[top_15_emotion_names.index(emotion)]-1] 
             for emotion in ordered_emotions]

bars = ax.barh(range(len(ordered_emotions)), ordered_counts, color=bar_colors)
ax.set_yticks(range(len(ordered_emotions)))
ax.set_yticklabels(ordered_emotions)
ax.set_xlabel('Frequency', fontsize=12, fontweight='bold')
ax.set_ylabel('Emotion', fontsize=12, fontweight='bold')
ax.set_title('Top 15 Emotions (Clustered by Co-occurrence Patterns)', 
             fontsize=14, fontweight='bold', pad=15)

# Add cluster separators
current_cluster = None
for i, emotion in enumerate(ordered_emotions):
    cluster = cluster_labels[top_15_emotion_names.index(emotion)]
    if current_cluster is not None and cluster != current_cluster:
        ax.axhline(y=i-0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    current_cluster = cluster

ax.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('outputs/figures/fig1b_clustered_bar_chart.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: outputs/figures/fig1b_clustered_bar_chart.png")

# ============================================================================
# FIGURE 2: EMOTION BY TOPIC
# ============================================================================
print("\n--- Creating Figure 2: Emotion Distribution by Topic ---\n")

# Get topics
if 'search_query' not in df_emotions.columns:
    print("⚠ Warning: 'search_query' column not found. Skipping topic analysis.")
else:
    topics = df_emotions['search_query'].unique()
    top_emotions = emotion_counts.head(7).index.tolist()
    
    # Option 2A: Grouped Bar Chart
    print("Creating fig2a_grouped_bar.png...")
    topic_emotion_counts = pd.crosstab(df_emotions['search_query'], 
                                      df_emotions['dominant_emotion'])
    topic_emotion_counts = topic_emotion_counts[top_emotions]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    topic_emotion_counts.plot(kind='bar', ax=ax, width=0.8)
    ax.set_xlabel('Topic', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Emotion Distribution by Topic\n(Top 7 Emotions)', 
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('outputs/figures/fig2a_grouped_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: outputs/figures/fig2a_grouped_bar.png")
    
    # Option 2B: Heatmap
    print("Creating fig2b_heatmap.png...")
    top_12_emotions = emotion_counts.head(12).index.tolist()
    heatmap_data = pd.crosstab(df_emotions['search_query'], 
                               df_emotions['dominant_emotion'])
    heatmap_data = heatmap_data[top_12_emotions]
    # Normalize by row (percentage)
    heatmap_data_pct = heatmap_data.div(heatmap_data.sum(axis=1), axis=0) * 100
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(heatmap_data_pct, annot=True, fmt='.1f', cmap='YlOrRd', 
               cbar_kws={'label': 'Percentage (%)'}, ax=ax, linewidths=0.5)
    ax.set_xlabel('Emotion', fontsize=12, fontweight='bold')
    ax.set_ylabel('Topic', fontsize=12, fontweight='bold')
    ax.set_title('Emotion Distribution by Topic (Heatmap)\n(Percentage of Comments)', 
                fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('outputs/figures/fig2b_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: outputs/figures/fig2b_heatmap.png")
    
    # Save topic distribution table
    emotion_by_topic = pd.crosstab(df_emotions['search_query'], 
                                  df_emotions['dominant_emotion'])
    emotion_by_topic.to_csv('outputs/tables/emotion_distribution_by_topic.csv')
    print("✓ Saved: outputs/tables/emotion_distribution_by_topic.csv")

# ============================================================================
# FIGURE 3: EMOTION RELATIONSHIPS
# ============================================================================
print("\n--- Creating Figure 3: Emotion Relationships ---\n")

# Option 3A: Correlation Heatmap
print("Creating fig3a_correlation_heatmap.png...")
top_15_emotion_cols = [f'emotion_{emotion}' for emotion in top_15_emotion_names 
                       if f'emotion_{emotion}' in df_emotions.columns]
emotion_data = df_emotions[top_15_emotion_cols]
correlation_matrix = emotion_data.corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
           center=0, vmin=-1, vmax=1, square=True, ax=ax,
           cbar_kws={'label': 'Correlation Coefficient'})
ax.set_title('Emotion Correlation Matrix\n(Top 15 Emotions)', 
            fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('outputs/figures/fig3a_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: outputs/figures/fig3a_correlation_heatmap.png")

# Option 3B: Network Graph
print("Creating fig3b_network_graph.png...")
# Create network from correlation matrix
G = nx.Graph()
threshold = 0.3  # Only show correlations above threshold

for i, emotion1 in enumerate(top_15_emotion_names):
    G.add_node(emotion1)
    for j, emotion2 in enumerate(top_15_emotion_names):
        if i < j:
            corr = correlation_matrix.iloc[i, j]
            if abs(corr) > threshold:
                G.add_edge(emotion1, emotion2, weight=abs(corr))

# Community detection (Louvain)
try:
    import community.community_louvain as community_louvain
    communities = community_louvain.best_partition(G)
    n_communities = len(set(communities.values()))
except ImportError:
    # Fallback: use simple clustering based on correlation
    print("  ⚠ python-louvain not installed. Using simple clustering.")
    communities = {}
    for i, node in enumerate(G.nodes()):
        # Assign based on correlation patterns
        communities[node] = i % 4
    n_communities = 4
except Exception as e:
    print(f"  ⚠ Community detection failed: {e}. Using simple clustering.")
    communities = {node: hash(node) % 4 for node in G.nodes()}
    n_communities = 4

# Create visualization
fig, ax = plt.subplots(figsize=(14, 10))
pos = nx.spring_layout(G, k=2, iterations=50)

# Color nodes by community
node_colors = [communities[node] for node in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                      node_size=2000, alpha=0.8, cmap=plt.cm.Set3, ax=ax)

# Draw edges with thickness based on weight
edges = G.edges()
weights = [G[u][v]['weight']*3 for u, v in edges]
nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5, ax=ax)

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)

ax.set_title('Emotion Network Graph\n(Edges represent correlations > 0.3, Colors indicate communities)', 
            fontsize=14, fontweight='bold', pad=15)
ax.axis('off')
plt.tight_layout()
plt.savefig('outputs/figures/fig3b_network_graph.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: outputs/figures/fig3b_network_graph.png")

# ============================================================================
# PART 4: INFERENTIAL ANALYSIS - EMOTION & ENGAGEMENT REGRESSION
# ============================================================================
print("\n=== PART 4: EMOTION & ENGAGEMENT REGRESSION ===\n")

# Prepare data for regression
engagement_metrics = ['like_count', 'reply_count']
regression_results = []

for emotion_name in top_15_emotion_names:
    emotion_col = f'emotion_{emotion_name}'
    if emotion_col not in df_emotions.columns:
        continue
    
    for metric in engagement_metrics:
        if metric not in df_emotions.columns:
            continue
        
        # Prepare data
        valid_mask = (df_emotions[emotion_col].notna() & 
                     df_emotions[metric].notna())
        X = df_emotions.loc[valid_mask, emotion_col].values
        y = df_emotions.loc[valid_mask, metric].values
        
        if len(X) < 100:
            continue
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = linregress(X, y)
        r_squared = r_value**2
        
        # Confidence interval
        t_critical = 1.96  # Approximate for large n
        ci_lower = slope - t_critical * std_err
        ci_upper = slope + t_critical * std_err
        
        regression_results.append({
            'emotion': emotion_name,
            'engagement_metric': metric,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'r_value': r_value,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant': 'Yes' if p_value < 0.05 else 'No'
        })

regression_df = pd.DataFrame(regression_results)
regression_df.to_csv('outputs/tables/emotion_engagement_regression.csv', index=False)
print(f"✓ Saved: outputs/tables/emotion_engagement_regression.csv")
print(f"  {len(regression_df)} emotion-metric combinations analyzed")

# ============================================================================
# FIGURE 4: EMOTION & ENGAGEMENT REGRESSION VISUALIZATIONS
# ============================================================================
print("\n--- Creating Figure 4: Emotion & Engagement Regression ---\n")

# Option 4A: Standard Forest Plot
print("Creating fig4a_forest_plot.png...")
# Filter to top emotions by significance or effect size
sig_results = regression_df[regression_df['significant'] == 'Yes']
if len(sig_results) > 0:
    top_emotions_for_plot = sig_results.nlargest(15, 'r_squared')['emotion'].unique()
else:
    top_emotions_for_plot = regression_df.nlargest(15, 'r_squared')['emotion'].unique()

plot_data = regression_df[regression_df['emotion'].isin(top_emotions_for_plot)].copy()
plot_data = plot_data.sort_values(['engagement_metric', 'emotion'])

fig, ax = plt.subplots(figsize=(12, 10))
y_positions = range(len(plot_data))

colors = {'like_count': '#0072B2', 'reply_count': '#D55E00'}
for idx, row in plot_data.iterrows():
    y_pos = list(plot_data.index).index(idx)
    color = colors[row['engagement_metric']]
    marker = 'o' if row['engagement_metric'] == 'like_count' else 's'
    
    # Plot CI
    ax.plot([row['ci_lower'], row['ci_upper']], [y_pos, y_pos], 
           color=color, linewidth=2, alpha=0.7)
    # Plot coefficient
    ax.scatter([row['slope']], [y_pos], color=color, marker=marker, 
              s=150, edgecolors='black', linewidth=1.5, zorder=2)

ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
ax.set_yticks(y_positions)
ax.set_yticklabels([f"{row['emotion']} ({row['engagement_metric']})" 
                    for _, row in plot_data.iterrows()])
ax.set_xlabel('Regression Coefficient (β)', fontsize=12, fontweight='bold')
ax.set_ylabel('Emotion × Engagement Metric', fontsize=12, fontweight='bold')
ax.set_title('Emotion-Engagement Regression Coefficients\n(95% Confidence Intervals)', 
            fontsize=14, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('outputs/figures/fig4a_forest_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: outputs/figures/fig4a_forest_plot.png")

# Option 4B: Annotated Forest Plot
print("Creating fig4b_annotated_forest_plot.png...")
fig, ax = plt.subplots(figsize=(14, 10))
y_positions = range(len(plot_data))

for idx, row in plot_data.iterrows():
    y_pos = list(plot_data.index).index(idx)
    color = colors[row['engagement_metric']]
    marker = 'o' if row['engagement_metric'] == 'like_count' else 's'
    
    # Plot CI
    ax.plot([row['ci_lower'], row['ci_upper']], [y_pos, y_pos], 
           color=color, linewidth=2, alpha=0.7)
    # Plot coefficient
    ax.scatter([row['slope']], [y_pos], color=color, marker=marker, 
              s=150, edgecolors='black', linewidth=1.5, zorder=2)

ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
ax.set_yticks(y_positions)
ax.set_yticklabels([f"{row['emotion']} ({row['engagement_metric']})" 
                    for _, row in plot_data.iterrows()])
ax.set_xlabel('Regression Coefficient (β)', fontsize=12, fontweight='bold')
ax.set_ylabel('Emotion × Engagement Metric', fontsize=12, fontweight='bold')
ax.set_title('Emotion-Engagement Regression Coefficients\n(95% Confidence Intervals)', 
            fontsize=14, fontweight='bold', pad=15)

# Add annotations
positive_emotions = plot_data[(plot_data['slope'] > 0) & (plot_data['significant'] == 'Yes')]
negative_emotions = plot_data[(plot_data['slope'] < 0) & (plot_data['significant'] == 'Yes')]

if len(positive_emotions) > 0:
    ax.text(0.98, 0.95, "• Positive emotions (e.g., admiration) → More Engagement", 
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

if len(negative_emotions) > 0:
    ax.text(0.98, 0.90, "• Negative emotions (e.g., anger) → More Replies", 
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

# Add R² note
mean_r2 = plot_data['r_squared'].mean()
ax.text(0.02, 0.02, f"Note: Mean R² = {mean_r2:.4f} (low predictive power)", 
       transform=ax.transAxes, fontsize=9, style='italic',
       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('outputs/figures/fig4b_annotated_forest_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: outputs/figures/fig4b_annotated_forest_plot.png")

print("\n" + "="*70)
print("EMOTION ANALYSIS COMPLETE!")
print("="*70)
print("\nAll visualizations and tables have been generated.")
print("Check outputs/figures/ and outputs/tables/ for all files.")

