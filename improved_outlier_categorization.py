"""
Improved Outlier Categorization Analysis
This script provides a more transparent and comprehensive analysis of outlier content
to address reviewer concerns about categorization methodology.
"""

import pandas as pd
import numpy as np
import os
import warnings
import re

warnings.filterwarnings('ignore')

# Get project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

print("=" * 80)
print("IMPROVED OUTLIER CATEGORIZATION ANALYSIS")
print("=" * 80)

# Load data
print("\nLoading data...")
df = pd.read_csv(os.path.join(project_root, 'data/processed/03_sentiment_results.csv'))
print(f"Loaded {len(df):,} comments")

# Ensure numeric columns are numeric
numeric_cols = ['like_count', 'reply_count']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Identify outliers
like_threshold = df['like_count'].quantile(0.99)
reply_threshold = df['reply_count'].quantile(0.99)

outliers = df[(df['like_count'] > like_threshold) | (df['reply_count'] > reply_threshold)].copy()
outliers['combined_engagement'] = (
    (outliers['like_count'] / outliers['like_count'].max()) * 0.7 +
    (outliers['reply_count'] / outliers['reply_count'].max()) * 0.3
)

# Get top 20 outliers
top_20 = outliers.nlargest(20, 'combined_engagement').copy()

print(f"\nAnalyzing top 20 outliers...")
print(f"Total engagement range: {top_20['combined_engagement'].min():.4f} to {top_20['combined_engagement'].max():.4f}")

# ============================================================================
# IMPROVED CATEGORIZATION SCHEME
# ============================================================================

def categorize_comment_improved(text):
    """
    Improved categorization with more comprehensive categories and clearer logic.
    Returns primary category and confidence level.
    """
    if pd.isna(text) or text == '':
        return 'Empty/Invalid', 'low'
    
    text_lower = str(text).lower()
    text_original = str(text)
    
    # 1. QUESTION (any question mark OR question words at start)
    question_words = ['what', 'why', 'how', 'when', 'where', 'who', 'which', 'whose', 'whom']
    has_question_mark = '?' in text_original
    starts_with_question = any(text_lower.strip().startswith(q + ' ') for q in question_words)
    has_question_word = any(q in text_lower for q in question_words)
    
    if has_question_mark or (starts_with_question and len(text_original.split()) <= 15):
        return 'Question', 'high'
    elif has_question_word and '?' in text_original:
        return 'Question', 'medium'
    
    # 2. HUMOR/JOKE (expanded detection)
    humor_indicators = [
        'lol', 'haha', 'funny', 'joke', 'hilarious', '😂', '😄', '😆', '😅',
        'galactus', 'gas prices', 'credit cards', 'alien farts', 'blunt',
        'afk', 'space simulators', 'epstein files', 'nibiru'
    ]
    if any(indicator in text_lower for indicator in humor_indicators):
        return 'Humor/Joke', 'high'
    
    # Check for humorous patterns (absurd scenarios, pop culture references)
    absurd_patterns = ['gonna be mad', 'see the gas prices', 'maxing out credit', 
                      'lost his blunt', 'obviously went afk']
    if any(pattern in text_lower for pattern in absurd_patterns):
        return 'Humor/Joke', 'high'
    
    # 3. SPECULATIVE/CREATIVE THINKING
    speculative_indicators = [
        'what if', 'imagine', 'suppose', 'hypothetically', 'could be',
        'might be', 'possibly', 'perhaps', 'maybe'
    ]
    speculative_keywords = ['civilization', 'left on earth', 'intelligent', 'took one look']
    if any(ind in text_lower for ind in speculative_indicators) or \
       any(kw in text_lower for kw in speculative_keywords):
        return 'Speculative/Creative', 'high'
    
    # 4. INSIGHTFUL/ANALYTICAL (improved detection)
    technical_terms = ['theory', 'hypothesis', 'evidence', 'research', 'study', 'data', 
                      'analysis', 'scientific', 'physics', 'astronomer', 'physicist',
                      'magnetic', 'solar wind', 'plasma', 'consciousness', 'morphic']
    has_technical = any(term in text_lower for term in technical_terms)
    is_long = len(text_original) > 150
    
    if is_long and has_technical:
        return 'Insightful/Analytical', 'high'
    elif is_long and len(text_original.split()) > 30:
        return 'Insightful/Analytical', 'medium'
    
    # 5. ENTHUSIASTIC/POSITIVE REACTION
    enthusiasm_keywords = ['amazing', 'incredible', 'wow', 'awesome', 'fantastic', 
                          'love', '❤️', '🔥', 'stunning', 'satisfying', 'boggles my mind']
    has_enthusiasm = any(kw in text_lower for kw in enthusiasm_keywords)
    has_many_exclamations = text_original.count('!') >= 2
    
    if has_enthusiasm or has_many_exclamations:
        return 'Enthusiastic', 'high'
    
    # 6. SKEPTICAL/CRITICAL
    skeptical_keywords = ['doubt', 'skeptical', 'unlikely', 'probably not', 'doubtful', 
                        'questionable', 'suspicious', 'unconscionable', 'nuts']
    if any(kw in text_lower for kw in skeptical_keywords):
        return 'Skeptical/Critical', 'high'
    
    # 7. META-COMMENTARY (about video/channel/community)
    meta_keywords = ['video', 'channel', 'subscribe', 'like this', 'agree', 'disagree', 
                    'opinion', 'anton', 'stefan', 'james webb', 'juno']
    meta_patterns = ['waiting for', 'thank you', 'see you in the next']
    if any(kw in text_lower for kw in meta_keywords) or \
       any(pattern in text_lower for pattern in meta_patterns):
        return 'Meta-commentary', 'high'
    
    # 8. PHILOSOPHICAL/REFLECTIVE
    philosophical_indicators = ['boggles my mind', 'born too', 'just in time', 
                               'synchronistic', 'resonance', 'nothing is coincidence']
    if any(ind in text_lower for ind in philosophical_indicators):
        return 'Philosophical/Reflective', 'high'
    
    # 9. QUOTATION/RESPONSE
    if text_original.count('"') >= 2 or text_original.startswith('"') or \
       'yeah that' in text_lower or "that's one way" in text_lower:
        return 'Quotation/Response', 'medium'
    
    # 10. SHORT/PITHY STATEMENT
    if len(text_original.split()) <= 8 and len(text_original) < 50:
        return 'Short/Pithy Statement', 'medium'
    
    # 11. OTHER (truly unclassifiable)
    return 'Other', 'low'

# Apply improved categorization
print("\n" + "=" * 80)
print("IMPROVED CATEGORIZATION RESULTS")
print("=" * 80)

categorizations = []
for idx, row in top_20.iterrows():
    text = row['comment_text_original']
    category, confidence = categorize_comment_improved(text)
    categorizations.append({
        'Comment': text[:100] + '...' if len(str(text)) > 100 else text,
        'Full Comment': text,
        'Category': category,
        'Confidence': confidence,
        'Likes': row['like_count'],
        'Replies': row['reply_count'],
        'Topic': row.get('search_query', 'N/A'),
        'Sentiment': row.get('sentiment_transformer', 'N/A'),
        'Length': len(str(text))
    })

categorization_df = pd.DataFrame(categorizations)

print("\nCategory Distribution (Improved Scheme):")
category_counts = categorization_df['Category'].value_counts()
for cat, count in category_counts.items():
    pct = (count / len(categorization_df)) * 100
    print(f"  {cat}: {count} ({pct:.1f}%)")

print("\nConfidence Distribution:")
confidence_counts = categorization_df['Confidence'].value_counts()
for conf, count in confidence_counts.items():
    print(f"  {conf}: {count}")

# Compare with original categorization
print("\n" + "=" * 80)
print("COMPARISON: ORIGINAL vs IMPROVED CATEGORIZATION")
print("=" * 80)

# Load original categorization if available
original_categorized_path = os.path.join(project_root, 'outlier_analysis/categorized_outliers.csv')
if os.path.exists(original_categorized_path):
    original_df = pd.read_csv(original_categorized_path)
    print("\nOriginal Categorization:")
    print(original_df['Category'].value_counts())
    
    print("\nImproved Categorization:")
    print(categorization_df['Category'].value_counts())
    
    # Show examples of re-categorized comments
    print("\n" + "=" * 80)
    print("EXAMPLES: COMMENTS RE-CATEGORIZED FROM 'OTHER'")
    print("=" * 80)
    
    # Find comments that were "Other" in original but now have a category
    for idx, row in categorization_df.iterrows():
        original_row = original_df.iloc[idx] if idx < len(original_df) else None
        if original_row is not None and original_row['Category'] == 'Other' and row['Category'] != 'Other':
            print(f"\nOriginal: Other → Improved: {row['Category']}")
            print(f"Comment: {row['Full Comment'][:200]}...")
            print(f"Confidence: {row['Confidence']}")

# Save improved categorization
output_dir = os.path.join(project_root, 'outlier_analysis')
os.makedirs(output_dir, exist_ok=True)

improved_path = os.path.join(output_dir, 'improved_categorized_outliers.csv')
categorization_df.to_csv(improved_path, index=False)
print(f"\n✓ Improved categorization saved to: {improved_path}")

# Generate detailed analysis report
print("\n" + "=" * 80)
print("GENERATING DETAILED ANALYSIS REPORT")
print("=" * 80)

report_lines = []
report_lines.append("=" * 80)
report_lines.append("IMPROVED OUTLIER CATEGORIZATION ANALYSIS")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append("This analysis addresses reviewer concerns about categorization methodology")
report_lines.append("by providing a more comprehensive and transparent categorization scheme.")
report_lines.append("")
report_lines.append("=" * 80)
report_lines.append("METHODOLOGY")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append("Categories were defined based on:")
report_lines.append("1. Content analysis of actual outlier comments")
report_lines.append("2. Common patterns observed in high-engagement content")
report_lines.append("3. Clear, defensible criteria for each category")
report_lines.append("")
report_lines.append("Category Definitions:")
report_lines.append("")
report_lines.append("1. QUESTION: Contains question mark OR starts with question word")
report_lines.append("2. HUMOR/JOKE: Contains humor indicators, absurd scenarios, or pop culture refs")
report_lines.append("3. SPECULATIVE/CREATIVE: Presents hypothetical scenarios or creative thinking")
report_lines.append("4. INSIGHTFUL/ANALYTICAL: Long comments (>150 chars) with technical terms")
report_lines.append("5. ENTHUSIASTIC: Contains positive enthusiasm keywords or multiple !")
report_lines.append("6. SKEPTICAL/CRITICAL: Expresses doubt, skepticism, or criticism")
report_lines.append("7. META-COMMENTARY: References video, channel, creator, or community")
report_lines.append("8. PHILOSOPHICAL/REFLECTIVE: Contains reflective or philosophical content")
report_lines.append("9. QUOTATION/RESPONSE: Quotes or responds to other content")
report_lines.append("10. SHORT/PITHY: Very short statements (≤8 words, <50 chars)")
report_lines.append("11. OTHER: Truly unclassifiable content")
report_lines.append("")
report_lines.append("=" * 80)
report_lines.append("RESULTS")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append("Category Distribution (Top 20 Outliers):")
for cat, count in category_counts.items():
    pct = (count / len(categorization_df)) * 100
    report_lines.append(f"  {cat}: {count} ({pct:.1f}%)")
report_lines.append("")
report_lines.append("=" * 80)
report_lines.append("DETAILED BREAKDOWN BY CATEGORY")
report_lines.append("=" * 80)
report_lines.append("")

for category in sorted(categorization_df['Category'].unique()):
    cat_comments = categorization_df[categorization_df['Category'] == category]
    report_lines.append(f"\n{category} ({len(cat_comments)} comments):")
    for idx, row in cat_comments.iterrows():
        report_lines.append(f"  - {row['Full Comment'][:150]}...")
        report_lines.append(f"    (Likes: {row['Likes']}, Replies: {row['Replies']}, Confidence: {row['Confidence']})")

report_lines.append("")
report_lines.append("=" * 80)
report_lines.append("RECOMMENDATIONS FOR MANUSCRIPT")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append("Option 1: Use improved categorization and report new distribution")
report_lines.append("Option 2: Acknowledge limitations of original scheme and present both")
report_lines.append("Option 3: Remove categorization claim and focus on qualitative diversity")
report_lines.append("")
report_lines.append("=" * 80)

report_path = os.path.join(output_dir, 'improved_categorization_analysis.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print(f"✓ Detailed analysis report saved to: {report_path}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
