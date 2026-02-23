# Reproducibility Index: Paper Figures and Tables

This document maps each figure and table in the paper *"Sentiment and emotion are weak predictors of engagement: A multi-method analysis of astrobiology topics on YouTube"* to the script(s) that generate them. Run scripts from the `Github_Copy` root directory.

## Prerequisites

1. Run the pipeline in order: `data_preparation` -> `exploratory_data_analysis` -> `preprocessing_and_feature_engineering` -> `sentiment_modeling` -> `downstream_analysis`
2. For emotion analysis (RQ4): Run `emotion_analysis.py` (requires `youtube_comments_with_emotions.csv` or emotion columns in processed data)

---

## Figures

| Paper | Description | Script | Output File |
|-------|-------------|--------|-------------|
| **Fig 1** | Research pipeline (conceptual) | N/A | Manual/diagram |
| **Fig 2** | Topic distribution (bar chart) | `exploratory_data_analysis.py` | `outputs/figures/topic_distribution.png` |
| **Fig 3** | Engagement distribution (4-panel: likes, replies, text length, word count) | `engagement_distribution_visualization.py` | `outputs/figures/engagement_distribution_skewness.png` |
| **Fig 4** | Temporal patterns (comment volume, avg likes/replies over time) | `distribution_and_timeseries_analysis.py` | `outputs/figures/timeseries_engagement.png` |
| **Fig 5** | Sentiment evolution over time + engagement by sentiment | `distribution_and_timeseries_analysis.py` | `outputs/figures/timeseries_sentiment.png` |
| **Fig 6** | Sentiment distributions across topics (stacked bars, 3 models) | `downstream_analysis.py` | `outputs/figures/sentiment_distribution_*.png` |
| **Fig 7** | Forest plot: sentiment regression coefficients | `forest_plot_visualization.py` | `outputs/figures/rq3_sentiment_engagement_regression_forest_enhanced.png` |
| **Fig 8** | Median engagement by sentiment (boxplots, log scale) | `downstream_analysis.py` | `outputs/figures/sentiment_engagement_relationship.png` |
| **Fig 9** | Correlation heatmaps (Spearman + Pearson) | `regenerate_correlation_heatmap_larger.py` | `outputs/figures/correlation_matrix.png` |
| **Fig 10** | Scatter: likes vs replies, outliers highlighted | `create_improved_scatter_plot.py` | `outputs/figures/likes_vs_replies_outliers.png` |
| **Fig 11** | Emotion valence diverging chart | `create_emotion_valence_diverging_chart.py` | `outputs/figures/emotion_diverging_chart.png` |
| **Fig 12** | Heatmap: top 20 emotions by topic | `emotion_analysis.py` | `outputs/figures/fig2b_heatmap.png` |
| **Fig 13** | Forest plot: emotions vs engagement | `emotion_analysis.py` | `outputs/figures/fig4a_forest_plot.png` |
| **Fig 14** | Decision framework (conceptual) | N/A | Manual/diagram |

---

## Tables

| Paper | Description | Script | Output File |
|-------|-------------|--------|-------------|
| **Table 1** | Features extracted from raw text | N/A | Documentation in `data/README.md` |
| **Table 2** | Three-track preprocessing strategy | N/A | Documentation in `utils/preprocessing.py` |
| **Table 3** | Summary of statistical tests | N/A | Methodology section |
| **Table 4** | Chi-square results (sentiment by topic) | `downstream_analysis.py` | `outputs/tables/chi_square_results.csv` |
| **Table 5** | Engagement patterns by topic (mean values) | `engagement_statistical_tests.py` | `outputs/tables/rq2_engagement_patterns_kruskal_wallis.csv` + `engagement_patterns_by_topic.csv` |
| **Table 6** | Mean engagement by topic (likes, replies) | `downstream_analysis.py` / `exploratory_data_analysis.py` | `outputs/tables/engagement_patterns_by_topic.csv` |
| **Table 7** | Linear regression: sentiment predicting engagement | `regression_analysis.py` | `outputs/tables/rq3_sentiment_engagement_regression.csv` |
| **Table 8** | Outliers vs typical comments comparison | `outlier_analysis.py` | `outputs/outlier_analysis/outlier_comparison_table.csv` |
| **Table 9** | Outlier characteristics by topic | `outlier_analysis.py` | `outputs/outlier_analysis/outlier_topic_analysis.csv` |
| **Table 10** | Qualitative examples of high-engagement comments | `outlier_analysis.py` | `outputs/outlier_analysis/top_outlier_examples.csv` |
| **Table 11** | Top 20 emotions overall | `emotion_analysis.py` | `outputs/emotion_analysis/tables/emotion_distribution_overall.csv` |
| **Table 12** | Emotions significantly associated with engagement | `emotion_analysis.py` | `outputs/emotion_analysis/tables/emotion_engagement_regression.csv` |
| **Table 13** | Method disagreement examples (3 comments) | `find_sentiment_disagreement_examples.py` | `outputs/tables/sentiment_method_disagreement_examples.csv` |

---

## Run Order for Full Reproducibility

```bash
# From Github_Copy/ directory
cd scripts

# 1. Pipeline (requires data/raw/ with YouTube comments)
python data_preparation.py
python exploratory_data_analysis.py
python preprocessing_and_feature_engineering.py
python sentiment_modeling.py
python downstream_analysis.py

# 2. RQ3-specific analyses
python regression_analysis.py
python forest_plot_visualization.py
python regenerate_correlation_heatmap_larger.py
python correlation_analysis.py

# 3. Engagement & distribution
python engagement_statistical_tests.py
python engagement_distribution_visualization.py
python distribution_and_timeseries_analysis.py
python test_normality_per_topic.py

# 4. Outlier analysis
python outlier_analysis.py
python extract_all_outliers.py

# 5. Sentiment disagreement examples (Table 13)
python find_sentiment_disagreement_examples.py

# 6. Scatter plot (Fig 10)
python create_improved_scatter_plot.py

# 7. Emotion analysis (requires emotion model; run after emotion_analysis.py)
python emotion_analysis.py
python create_emotion_valence_diverging_chart.py
python create_all_27_emotions_figure_and_table.py
python chi_square_emotion_topic_test.py
python emotion_outlier_analysis.py
```

---

## Data Path Notes

- Scripts expect `data/processed/03_sentiment_results.csv` (or `02_preprocessed_data.csv` for earlier steps)
- Use `data/sample_data.csv` for quick testing (100 rows); full results require the complete dataset
- Emotion scripts require `youtube_comments_with_emotions.csv` or emotion columns in the sentiment results
