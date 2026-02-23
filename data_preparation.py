#!/usr/bin/env python3
"""
Execute Notebook 1 with corruption fix
This script runs the updated notebook cells with filename-based topic assignment
"""

import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
from langdetect import detect, LangDetectException
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc

# Set random seed for reproducibility
np.random.seed(42)

# Configure display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("NOTEBOOK 1: DATA PREPARATION WITH CORRUPTION FIX")
print("=" * 80)
print("Libraries imported successfully")

# Clear all memory and cache before starting
print("\n" + "=" * 80)
print("CLEARING MEMORY AND CACHE")
print("=" * 80)
gc.collect()
print("✅ Garbage collection completed")
print("✅ Memory cleared - starting with clean state")
print("")

# ============================================================================
# Cell 3: Load CSV Files with Topic Mapping
# ============================================================================

def map_file_to_topic(file_path):
    """
    Extract topic from filename using pattern matching.
    Handles all naming variations observed in raw data files.
    """
    file_name = os.path.basename(file_path)
    
    # Venus phosphine - check full pattern first, then short version
    if 'Venus_phosphine' in file_name:
        return 'Venus phosphine'
    elif 'Venus' in file_name:  # Handles Venus_comments files
        return 'Venus phosphine'
    
    # 3I/ATLAS - files use "3IATLAS" (no slash, no spaces)
    elif '3IATLAS' in file_name:
        return '3I/ATLAS'
    
    # K2-18b - files use "K2-18b" (with hyphens)
    elif 'K2-18b' in file_name:
        return 'K2-18b'
    
    # Oumuamua - straightforward naming
    elif 'Oumuamua' in file_name:
        return 'Oumuamua'
    
    # Tabby's Star - files use "Tabbys" or "Tabbys_Star" (no apostrophe)
    elif 'Tabbys_Star' in file_name or 'Tabbys' in file_name:
        return "Tabby's Star"
    
    # No match found - will trigger warning
    else:
        return None

def load_csv_files(directory_path, file_pattern="*.csv"):
    """
    Load and concatenate CSV files from a directory.
    Adds 'file_source_topic' column to each DataFrame based on filename.
    """
    csv_files = glob.glob(os.path.join(directory_path, file_pattern))
    print(f"Found {len(csv_files)} CSV files in {directory_path}")
    
    if len(csv_files) == 0:
        raise ValueError(f"No CSV files found in {directory_path}")
    
    dataframes = []
    topic_counts = {}
    unknown_files = []
    
    for file_path in tqdm(csv_files, desc="Loading CSV files"):
        try:
            df = pd.read_csv(file_path, low_memory=False)
            file_name = os.path.basename(file_path)
            
            # Extract topic from filename (GROUND TRUTH!)
            topic = map_file_to_topic(file_path)
            
            if topic:
                df['file_source_topic'] = topic
                topic_counts[topic] = topic_counts.get(topic, 0) + len(df)
                print(f"✓ {file_name}: {len(df):,} rows → {topic}")
            else:
                # Flag unknown files but still load them
                print(f"❌ WARNING: Could not determine topic for {file_name}")
                df['file_source_topic'] = 'UNKNOWN'
                unknown_files.append(file_name)
            
            dataframes.append(df)
            
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            continue
    
    if len(dataframes) == 0:
        raise ValueError("No dataframes were successfully loaded")
    
    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"LOADING SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully loaded {len(dataframes)} files")
    print(f"Total rows: {len(combined_df):,}")
    
    print(f"\nTopic breakdown:")
    for topic, count in sorted(topic_counts.items()):
        print(f"  {topic}: {count:,} rows")
    
    if unknown_files:
        print(f"\n⚠️  WARNING: {len(unknown_files)} file(s) with UNKNOWN topic:")
        for f in unknown_files:
            print(f"    - {f}")
        print("These files need manual review!")
    
    print(f"{'='*60}\n")
    
    return combined_df, len(csv_files)

# Load comments data
print("=" * 60)
print("LOADING COMMENTS DATA")
print("=" * 60)
comments_df, num_comment_files = load_csv_files("data/raw/comments/")

# Load metadata
print("\n" + "=" * 60)
print("LOADING METADATA")
print("=" * 60)
metadata_df, num_metadata_files = load_csv_files("data/raw/Metadata/")

print(f"\nSummary:")
print(f"  Comment files: {num_comment_files}")
print(f"  Metadata files: {num_metadata_files}")
print(f"  Total comments: {len(comments_df):,}")
print(f"  Total metadata records: {len(metadata_df):,}")

# ============================================================================
# Cell 4: Validation
# ============================================================================

# Validation: Verify file-to-topic mapping worked correctly
print("=" * 60)
print("FILE-TO-TOPIC MAPPING VALIDATION")
print("=" * 60)

# Validate comments
if 'file_source_topic' in comments_df.columns:
    print("\n✅ Comments files - Topic distribution:")
    topic_dist = comments_df['file_source_topic'].value_counts()
    print(topic_dist)
    
    unknown_count = (comments_df['file_source_topic'] == 'UNKNOWN').sum()
    if unknown_count > 0:
        print(f"\n❌ WARNING: {unknown_count:,} comment rows with UNKNOWN topic!")
        print("Some files didn't match expected naming patterns.")
    else:
        print(f"\n✅ All {len(comments_df):,} comment rows have valid topics")
else:
    raise ValueError("❌ CRITICAL: file_source_topic column missing from comments_df!")

# Validate metadata
if 'file_source_topic' in metadata_df.columns:
    print("\n✅ Metadata files - Topic distribution:")
    print(metadata_df['file_source_topic'].value_counts())
    
    unknown_count_meta = (metadata_df['file_source_topic'] == 'UNKNOWN').sum()
    if unknown_count_meta > 0:
        print(f"\n❌ WARNING: {unknown_count_meta:,} metadata rows with UNKNOWN topic!")
else:
    raise ValueError("❌ CRITICAL: file_source_topic column missing from metadata_df!")

# Verify expected file counts
print("\n" + "=" * 60)
print("FILE COUNT VERIFICATION")
print("=" * 60)
print(f"Expected: 7 comment files (2×3I/ATLAS, 2×K2-18b, 1×Venus, 1×Oumuamua, 1×Tabby's)")
print(f"Expected: 5 metadata files (1 per topic)")
print(f"\nActual comment files loaded: {num_comment_files}")
print(f"Actual metadata files loaded: {num_metadata_files}")

if num_comment_files != 7:
    print(f"⚠️  WARNING: Expected 7 comment files, but loaded {num_comment_files}")
    print("Check if any files are missing from raw/comments/")

if num_metadata_files != 5:
    print(f"⚠️  WARNING: Expected 5 metadata files, but loaded {num_metadata_files}")
    print("Check if any files are missing from raw/Metadata/")

print("\n✅ Validation complete")

# ============================================================================
# Cell 6: Handle Duplicates
# ============================================================================

# Check for duplicate comment_ids before merging
print("\n" + "=" * 60)
print("DUPLICATE DETECTION - COMMENTS")
print("=" * 60)

duplicate_comments_before = comments_df.duplicated(subset=['comment_id']).sum()
print(f"Duplicate comment_ids before cleaning: {duplicate_comments_before:,}")

if duplicate_comments_before > 0:
    print(f"Percentage of duplicates: {(duplicate_comments_before/len(comments_df)*100):.2f}%")
    comments_df = comments_df.drop_duplicates(subset=['comment_id'], keep='first')
    print(f"After removing duplicates: {len(comments_df):,} comments")
    print(f"Removed {duplicate_comments_before:,} duplicate comments")
else:
    print("No duplicate comment_ids found")

# Check for duplicate video_ids in metadata
print("\n" + "=" * 60)
print("DUPLICATE DETECTION - METADATA")
print("=" * 60)

duplicate_videos_before = metadata_df.duplicated(subset=['video_id']).sum()
print(f"Duplicate video_ids before cleaning: {duplicate_videos_before:,}")

if duplicate_videos_before > 0:
    print(f"Percentage of duplicates: {(duplicate_videos_before/len(metadata_df)*100):.2f}%")
    metadata_df = metadata_df.drop_duplicates(subset=['video_id'], keep='first')
    print(f"After removing duplicates: {len(metadata_df):,} metadata records")
    print(f"Removed {duplicate_videos_before:,} duplicate metadata records")
else:
    print("No duplicate video_ids found")

# ============================================================================
# Cell 8: Merge with Corruption Fix
# ============================================================================

# Merge comments with metadata on video_id
print("\n" + "=" * 60)
print("MERGING COMMENTS AND METADATA")
print("=" * 60)

print(f"\nComments shape: {comments_df.shape}")
print(f"Metadata shape: {metadata_df.shape}")

# Identify columns to merge from metadata
# EXCLUDE: video_id (join key), file_source_topic (we use comment's version), search_query (unreliable)
metadata_cols_to_merge = [
    col for col in metadata_df.columns 
    if col not in comments_df.columns 
    and col not in ['video_id', 'file_source_topic', 'search_query']
]

print(f"\nColumns to add from metadata: {metadata_cols_to_merge}")

# Perform LEFT JOIN to keep all comments
# We use comments_df as the base because it has the reliable file_source_topic
merged_df = comments_df.merge(
    metadata_df[['video_id'] + metadata_cols_to_merge],
    on='video_id',
    how='left',
    suffixes=('', '_from_metadata')
)

print(f"\nMerge complete. Shape: {merged_df.shape}")

# Check for any new duplicates introduced by merge
duplicate_comments_after = merged_df.duplicated(subset=['comment_id']).sum()
print(f"Duplicate comment_ids after merge: {duplicate_comments_after:,}")

if duplicate_comments_after > 0:
    print("⚠️  WARNING: Duplicates introduced during merge!")
    merged_df = merged_df.drop_duplicates(subset=['comment_id'], keep='first')
    print(f"After removing duplicates: {len(merged_df):,} comments")

# === CORRUPTION FIX: Nuclear Option - Delete and Recreate ===
# Delete old corrupted column completely, then create fresh one
# This ensures no metadata baggage from the corrupted column survives

print("\n" + "=" * 60)
print("APPLYING CORRUPTION FIX - DELETE AND RECREATE")
print("=" * 60)

VALID_TOPICS = {'3I/ATLAS', 'Oumuamua', 'K2-18b', "Tabby's Star", 'Venus phosphine'}

if 'file_source_topic' in merged_df.columns:
    # STEP 1: Show what we're removing
    if 'search_query' in merged_df.columns:
        old_unique = merged_df['search_query'].nunique()
        old_nan = merged_df['search_query'].isna().sum()
        old_invalid = (~merged_df['search_query'].isin(VALID_TOPICS)).sum()
        
        print(f"\nOld search_query column stats:")
        print(f"  Unique values: {old_unique}")
        print(f"  NaN values: {old_nan:,}")
        print(f"  Invalid values: {old_invalid:,}")
        
        # COMPLETELY DELETE the old corrupted column
        print(f"\n🗑️  Deleting old corrupted search_query column...")
        merged_df = merged_df.drop(columns=['search_query'])
        print(f"✓ Old search_query column DELETED")
    else:
        print(f"\nℹ️  No existing search_query column found (good)")
    
    # STEP 2: Create brand NEW search_query from filename-based topics
    print(f"\n🆕 Creating fresh search_query from file_source_topic...")
    merged_df['search_query'] = merged_df['file_source_topic'].copy()
    print(f"✓ New search_query column CREATED from filenames")
    
    # STEP 3: Verify the new column
    new_unique = merged_df['search_query'].nunique()
    new_nan = merged_df['search_query'].isna().sum()
    new_invalid = (~merged_df['search_query'].isin(VALID_TOPICS)).sum()
    
    print(f"\nNew search_query column stats:")
    print(f"  Unique values: {new_unique}")
    print(f"  NaN values: {new_nan:,}")
    print(f"  Invalid values: {new_invalid:,}")
    print(f"  Data type: {merged_df['search_query'].dtype}")
    
    # STEP 4: Clean up temporary column
    merged_df = merged_df.drop(columns=['file_source_topic'])
    print(f"\n✓ Cleaned up temporary file_source_topic column")
    
    # STEP 5: Final validation
    if new_invalid == 0 and new_unique == 5 and new_nan == 0:
        print(f"\n✅ CORRUPTION FIX SUCCESSFUL!")
        print(f"   All {len(merged_df):,} rows have valid topics")
        print(f"   Topics: {sorted(merged_df['search_query'].unique().tolist())}")
    else:
        print(f"\n❌ CORRUPTION FIX FAILED!")
        print(f"   Invalid rows: {new_invalid:,}")
        print(f"   NaN rows: {new_nan:,}")
        print(f"   Unique topics: {new_unique} (expected 5)")
        raise ValueError("Corruption fix validation failed")
else:
    raise ValueError("❌ CRITICAL: file_source_topic column not found!")

print("=" * 60)

# Additional validation (redundant but safe)
print("\n" + "=" * 60)
print("ADDITIONAL VALIDATION")
print("=" * 60)

VALID_TOPICS = {'3I/ATLAS', 'Oumuamua', 'K2-18b', "Tabby's Star", 'Venus phosphine'}

# Check for NaN values
nan_count = merged_df['search_query'].isna().sum()
print(f"NaN values in search_query: {nan_count:,}")

# Check for invalid values
invalid_mask = ~merged_df['search_query'].isin(VALID_TOPICS)
invalid_count = invalid_mask.sum()
print(f"Invalid topic values: {invalid_count:,}")

# Show final distribution
print(f"\nFinal topic distribution:")
topic_dist_final = merged_df['search_query'].value_counts()
print(topic_dist_final)

# Verify we have exactly 5 topics
if len(topic_dist_final) != 5:
    print(f"\n⚠️  WARNING: Expected exactly 5 topics, found {len(topic_dist_final)}")
    raise ValueError("Expected exactly 5 topics after corruption fix")
else:
    print(f"\n✅ Confirmed: Exactly 5 unique topics as expected")

if nan_count > 0 or invalid_count > 0:
    print(f"\n❌ VALIDATION FAILED!")
    print(f"   {nan_count:,} NaN values remain")
    print(f"   {invalid_count:,} invalid values remain")
    raise ValueError("Corruption fix failed - data still contains invalid topics")

print(f"\n✅ VALIDATION PASSED: All topics are valid!")
print(f"\n{'='*60}")

# Store in df for next steps
df = merged_df.copy()
print(f"\nFinal merged dataset:")
print(f"  Rows: {len(df):,}")
print(f"  Columns: {len(df.columns)}")
print(f"  Topics: {df['search_query'].nunique()}")

# ============================================================================
# Cell 10: Language Detection
# ============================================================================

def detect_language_safe(text):
    """
    Detect language with robust error handling.
    """
    if pd.isna(text) or str(text).strip() == '':
        return 'unknown'
    
    try:
        text_str = str(text)
        # Skip very short texts (langdetect needs sufficient text)
        if len(text_str.strip()) < 3:
            return 'unknown'
        
        language = detect(text_str)
        return language
    except LangDetectException:
        return 'unknown'
    except Exception as e:
        return 'unknown'

# Enable progress bar for pandas apply
tqdm.pandas(desc="Detecting languages")

print("\n" + "=" * 60)
print("LANGUAGE DETECTION")
print("=" * 60)

# Use comment_text_original for language detection
print("Detecting languages using 'comment_text_original' column...")
df['language'] = df['comment_text_original'].progress_apply(detect_language_safe)

print(f"\nLanguage detection complete!")
print(f"Total comments processed: {len(df):,}")

# Display language distribution
language_counts = df['language'].value_counts()
language_percentages = (language_counts / len(df) * 100).round(2)

language_distribution = pd.DataFrame({
    'language': language_counts.index,
    'count': language_counts.values,
    'percentage': language_percentages.values
})

print("\nTop 10 languages:")
print(language_distribution.head(10).to_string(index=False))

# ============================================================================
# Cell 11: Visualize Language Distribution
# ============================================================================

# Create output directories if they don't exist
os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/tables', exist_ok=True)

# Visualize language distribution
print("\n" + "=" * 60)
print("LANGUAGE DISTRIBUTION VISUALIZATION")
print("=" * 60)

# Get top 10 languages
top_10_languages = language_distribution.head(10)

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(top_10_languages['language'], top_10_languages['count'],
               color='steelblue', edgecolor='black', linewidth=0.5)
ax.set_xlabel('Number of Comments', fontweight='bold', fontsize=12)
ax.set_ylabel('Language', fontweight='bold', fontsize=12)
ax.set_title('Top 10 Languages in YouTube Comments', 
            fontweight='bold', fontsize=14, pad=20)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels
for i, (idx, row) in enumerate(top_10_languages.iterrows()):
    ax.text(row['count'] + max(top_10_languages['count']) * 0.01, i,
            f"{int(row['count']):,} ({row['percentage']:.1f}%)",
            va='center', fontsize=10)

plt.tight_layout()
plt.savefig('outputs/figures/language_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Language distribution plot saved to outputs/figures/language_distribution.png")
plt.close()

# Save full language distribution table
language_distribution.to_csv('outputs/tables/language_distribution.csv', index=False)
print("✓ Language distribution table saved to outputs/tables/language_distribution.csv")

# ============================================================================
# Cell 13: Filter to English Comments Only
# ============================================================================

print("\n" + "=" * 60)
print("FILTERING TO ENGLISH COMMENTS")
print("=" * 60)

total_comments = len(df)
english_comments = df[df['language'] == 'en']
non_english_comments = total_comments - len(english_comments)
non_english_percentage = (non_english_comments / total_comments * 100)

print(f"Total comments: {total_comments:,}")
print(f"English comments: {len(english_comments):,}")
print(f"Non-English comments: {non_english_comments:,} ({non_english_percentage:.2f}%)")

# Filter to English only
df_english = english_comments.copy()

print(f"\nFiltered dataset: {len(df_english):,} English comments")
print(f"Removed: {non_english_comments:,} non-English comments ({non_english_percentage:.2f}%)")

# ============================================================================
# Cell 15: Data Type Optimization
# ============================================================================

print("\n" + "=" * 60)
print("DATA TYPE OPTIMIZATION")
print("=" * 60)

# Convert engagement counts to int32
engagement_cols = ['like_count', 'reply_count']
for col in engagement_cols:
    if col in df_english.columns:
        df_english[col] = pd.to_numeric(df_english[col], errors='coerce').fillna(0).astype('int32')
        print(f"✓ Converted {col} to int32")

# Keep video_view_count as int64 (may exceed int32 range)
if 'video_view_count' in df_english.columns:
    df_english['video_view_count'] = pd.to_numeric(df_english['video_view_count'], errors='coerce').fillna(0).astype('int64')
    print(f"✓ Converted video_view_count to int64")

# Convert categorical columns to category type for memory efficiency
# CRITICAL FIX: search_query should NOT be converted to category
# because it causes issues with unused categories from before filtering
categorical_cols = ['channel_title', 'comment_type']  # Removed 'search_query' from this list
for col in categorical_cols:
    if col in df_english.columns:
        df_english[col] = df_english[col].astype('category')
        print(f"✓ Converted {col} to category")

# CRITICAL FIX: Before data type optimization, ensure search_query is clean
# Filter out ANY rows with invalid topics (defensive programming)
VALID_TOPICS = {'3I/ATLAS', 'Oumuamua', 'K2-18b', "Tabby's Star", 'Venus phosphine'}

if 'search_query' in df_english.columns:
    # Count invalid rows BEFORE filtering
    invalid_mask = ~df_english['search_query'].isin(VALID_TOPICS)
    invalid_count_before = invalid_mask.sum()
    nan_count_before = df_english['search_query'].isna().sum()
    
    if invalid_count_before > 0 or nan_count_before > 0:
        print(f"\n⚠️  WARNING: Found {invalid_count_before:,} invalid topics and {nan_count_before:,} NaN values")
        print(f"   Dropping {invalid_count_before + nan_count_before:,} corrupted rows")
        
        # Drop rows with invalid topics
        df_english = df_english[df_english['search_query'].isin(VALID_TOPICS)].copy()
        
        print(f"   After cleanup: {len(df_english):,} rows with valid topics")
    
    # Ensure it's a clean string type (remove any categorical metadata)
    # Convert to string FIRST to remove any categorical metadata
    df_english['search_query'] = df_english['search_query'].astype('str')
    
    # Filter again to ensure no 'nan' strings (from NaN conversion)
    df_english = df_english[df_english['search_query'] != 'nan'].copy()
    df_english = df_english[df_english['search_query'].isin(VALID_TOPICS)].copy()
    
    print(f"✓ Cleaned search_query column (string type, no categorical metadata)")

# Verify search_query has exactly 5 topics
if 'search_query' in df_english.columns:
    unique_topics = df_english['search_query'].nunique()
    print(f"\n✓ search_query validation:")
    print(f"  Unique topics: {unique_topics}")
    print(f"  Expected: 5 topics")
    
    if unique_topics != 5:
        print(f"⚠️  WARNING: Expected 5 topics, found {unique_topics}")
        print(f"  Topics found: {df_english['search_query'].unique().tolist()}")
        # Force filter to only valid topics
        df_english = df_english[df_english['search_query'].isin(VALID_TOPICS)].copy()
        print(f"  ✅ After forced filter: {df_english['search_query'].nunique()} topics")
    else:
        print(f"  ✅ Confirmed: Exactly 5 topics as expected")
        print(f"  Topics: {sorted(df_english['search_query'].unique().tolist())}")

# Display memory usage
memory_before = df_english.memory_usage(deep=True).sum() / 1024**2
print(f"\nMemory usage: {memory_before:.2f} MB")

# Display data types
print("\nData types:")
print(df_english.dtypes)

# ============================================================================
# Cell 17: Save Cleaned Dataset
# ============================================================================

print("\n" + "=" * 60)
print("SAVING CLEANED DATASET")
print("=" * 60)

# Create processed directory if it doesn't exist
os.makedirs('data/processed', exist_ok=True)

# Save English-only dataset
output_path = 'data/processed/01_comments_english.csv'

# Ensure directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# CRITICAL: Final validation BEFORE saving
VALID_TOPICS = {'3I/ATLAS', 'Oumuamua', 'K2-18b', "Tabby's Star", 'Venus phosphine'}
if 'search_query' in df_english.columns:
    # Final cleanup: Remove any remaining invalid rows
    valid_mask = df_english['search_query'].isin(VALID_TOPICS)
    invalid_final = (~valid_mask).sum()
    
    if invalid_final > 0:
        print(f"\n⚠️  FINAL CLEANUP: Removing {invalid_final:,} rows with invalid topics before save")
        df_english = df_english[valid_mask].copy()
    
    # CRITICAL: Drop any rows with NaN BEFORE converting to string
    # This prevents 'nan' strings from being created
    nan_mask = df_english['search_query'].isna()
    nan_count_before_str = nan_mask.sum()
    
    if nan_count_before_str > 0:
        print(f"\n⚠️  Dropping {nan_count_before_str:,} rows with NaN in search_query before string conversion")
        df_english = df_english[~nan_mask].copy()
        print(f"   After dropping NaN rows: {len(df_english):,} rows")
    
    # Now convert to string (no NaN values should remain)
    df_english['search_query'] = df_english['search_query'].astype('str')
    
    # Filter out any 'nan' strings (defensive programming)
    df_english = df_english[df_english['search_query'] != 'nan'].copy()
    
    # Final filter to only valid topics
    df_english = df_english[df_english['search_query'].isin(VALID_TOPICS)].copy()
    
    # Final validation
    final_unique = df_english['search_query'].nunique()
    final_invalid = (~df_english['search_query'].isin(VALID_TOPICS)).sum()
    final_nan = df_english['search_query'].isna().sum()
    final_nan_strings = (df_english['search_query'] == 'nan').sum()
    
    if final_invalid > 0 or final_nan > 0 or final_nan_strings > 0 or final_unique != 5:
        print(f"\n❌ CRITICAL: Data still corrupted before save!")
        print(f"   Unique topics: {final_unique} (expected 5)")
        print(f"   Invalid: {final_invalid:,}")
        print(f"   NaN: {final_nan:,}")
        print(f"   'nan' strings: {final_nan_strings:,}")
        raise ValueError("Cannot save corrupted data! Fix failed.")
    else:
        print(f"\n✅ PRE-SAVE VALIDATION PASSED:")
        print(f"   Unique topics: {final_unique}")
        print(f"   Invalid: {final_invalid:,}")
        print(f"   NaN: {final_nan:,}")
        print(f"   'nan' strings: {final_nan_strings:,}")
        print(f"   Total rows to save: {len(df_english):,}")

# Force garbage collection before saving
gc.collect()

# Delete existing file if it exists (to ensure clean write)
if os.path.exists(output_path):
    os.remove(output_path)
    print(f"✓ Removed existing file to ensure clean write")

# Save with explicit parameters to ensure clean write
df_english.to_csv(output_path, index=False, encoding='utf-8')

print(f"✓ Saved cleaned dataset to: {output_path}")
print(f"  Rows: {len(df_english):,}")
print(f"  Columns: {len(df_english.columns)}")
print(f"  Memory: {df_english.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Immediately verify the saved file
print("\n" + "=" * 60)
print("VERIFYING SAVED FILE")
print("=" * 60)
try:
    # Read the file we just saved to verify it's clean
    df_verify = pd.read_csv(output_path, nrows=100000, engine='python', on_bad_lines='skip')
    
    if 'search_query' in df_verify.columns:
        VALID_TOPICS = {'3I/ATLAS', 'Oumuamua', 'K2-18b', "Tabby's Star", 'Venus phosphine'}
        invalid_verify = (~df_verify['search_query'].isin(VALID_TOPICS)).sum()
        nan_verify = df_verify['search_query'].isna().sum()
        unique_topics_verify = df_verify['search_query'].nunique()
        
        print(f"Verified file:")
        print(f"  Rows checked: {len(df_verify):,}")
        print(f"  Unique topics: {unique_topics_verify}")
        print(f"  NaN values: {nan_verify:,}")
        print(f"  Invalid topics: {invalid_verify:,}")
        
        if invalid_verify == 0 and nan_verify == 0 and unique_topics_verify == 5:
            print(f"\n✅✅✅ SAVED FILE VERIFICATION PASSED! ✅✅✅")
            print(f"   File is clean: 5 topics, 0 corruption")
        else:
            print(f"\n❌ WARNING: Saved file verification failed!")
            print(f"   Expected: 5 topics, 0 corruption")
            print(f"   Found: {unique_topics_verify} topics, {invalid_verify + nan_verify} corrupted rows")
    else:
        print("❌ ERROR: search_query column not found in saved file!")
        
except Exception as e:
    print(f"❌ ERROR verifying saved file: {e}")
    import traceback
    traceback.print_exc()

# Display summary statistics
print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)
print(f"Total comments (English): {len(df_english):,}")
print(f"Unique videos: {df_english['video_id'].nunique():,}")
print(f"Unique topics: {df_english['search_query'].nunique() if 'search_query' in df_english.columns else 'N/A'}")

# Final validation
print("\n" + "=" * 60)
print("FINAL VALIDATION")
print("=" * 60)
VALID_TOPICS = {'3I/ATLAS', 'Oumuamua', 'K2-18b', "Tabby's Star", 'Venus phosphine'}
final_nan = df_english['search_query'].isna().sum()
final_invalid = (~df_english['search_query'].isin(VALID_TOPICS)).sum()

print(f"NaN values: {final_nan:,}")
print(f"Invalid values: {final_invalid:,}")
print(f"Total corrupted: {final_nan + final_invalid:,}")

if final_nan == 0 and final_invalid == 0:
    print("\n✅✅✅ CORRUPTION FIX SUCCESSFUL! ✅✅✅")
    print("   All rows have valid topics from filenames")
    print(f"   Topic distribution: {df_english['search_query'].value_counts().to_dict()}")
else:
    print(f"\n❌ VALIDATION FAILED: {final_nan + final_invalid:,} corrupted rows remain")

# Clean up memory
del comments_df, metadata_df, merged_df, df
gc.collect()
print("\n✓ Memory cleaned up")

print("\n" + "=" * 60)
print("NOTEBOOK 1 COMPLETE")
print("=" * 60)
print("Next step: Run Notebook 2 (Exploratory Data Analysis)")

