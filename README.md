# Data Directory

## YouTube API Data Structure

This analysis uses YouTube comment data collected via the YouTube Data API v3. The data structure follows the API's comment resource schema.

### Raw Data Format

**Comments CSV** (`data/raw/comments/*.csv`):
- `comment_id`: Unique identifier for each comment
- `comment_text_original`: Original comment text (YouTube API `textOriginal`)
- `video_id`: YouTube video identifier
- `search_query`: Astrobiology topic used to collect videos (one of: 3I/ATLAS, Oumuamua, K2-18b, Tabby's Star, Venus phosphine)
- `like_count`: Number of likes on the comment
- `reply_count`: Number of replies to the comment
- `published_at`: Comment publication timestamp (ISO format)
- `author_name`, `author_channel_id`: Comment author metadata

**Metadata CSV** (`data/raw/Metadata/*.csv`):
- `video_id`, `search_query`: Video-topic mapping
- `view_count`, `title`, `channel_title`: Video metadata

### Processed Data Pipeline

1. **01_comments_english.csv**: Cleaned, English-only dataset after language detection
2. **02_preprocessed_data.csv**: Four text tracks (TextBlob, VADER, Transformer, Raw) + engagement features
3. **03_sentiment_results.csv**: Full dataset with sentiment labels from TextBlob, VADER, and Transformer models

### Sample Data

`sample_data.csv` contains 100 rows illustrating the full processed schema including sentiment and emotion columns. Use this for testing scripts without the full dataset.

### Obtaining Full Data

**Processed dataset (recommended):** Available on Zenodo — [10.5281/zenodo.18738200](https://doi.org/10.5281/zenodo.18738200)

Includes: `01_comments_english.csv`, `02_preprocessed_data.csv`, `03_sentiment_results.csv`, `youtube_comments_with_emotions.csv`

**Raw data collection:** The full dataset was collected using the YouTube Data API. Researchers can replicate data collection by:
1. Using the same search queries (astrobiology topics)
2. Extracting comments via the `commentThreads.list` and `comments.list` endpoints
3. Storing results in the CSV structure described above
