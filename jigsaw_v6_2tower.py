#%%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#%%
# All imports needed
import re
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import torch

from transformers import AutoModelForSequenceClassification, XLMRobertaTokenizer #DebertaV2Tokenizer # AutoTokenizer
from transformers import get_linear_schedule_with_warmup

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from tqdm import tqdm # For progress bars in console

import google.protobuf
#%%
# Cleans and tokenizes text, preserving non-English words and symbols.
def clean_and_tokenize_multilingual(text):
    if not isinstance(text, str):
        return []

    # 1. Lowercase the text
    text = text.lower()

    # 2. Add spaces around any character that is NOT a letter, number, or whitespace.
    # This isolates punctuation, emojis, and other symbols as separate tokens.
    # For example, "I hate you!!!" becomes " i hate you !!! "
    text = re.sub(r'([^a-zA-Z0-9\s])', r' \1 ', text)

    # 3. Split text into words (tokenize)
    words = text.split()

    # 4. Remove English stop words and short words
    # We keep non-English words because they are not in the stop_words list.
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    return keywords
#%%
# Create a set of stop words for faster lookup (ensure this is defined once)
stop_words = set(ENGLISH_STOP_WORDS)
#%%
def clean_and_tokenize_with_urls(text):
    """
    Cleans and tokenizes text, preserving non-English words and symbols,
    and treating full URLs as single keywords.
    """
    if not isinstance(text, str):
        return []

    # 1. Find and extract URLs
    # This regex captures http(s):// followed by non-whitespace, or www. followed by non-whitespace
    url_pattern = r'https?://\S+|www\.\S+'
    urls = re.findall(url_pattern, text)

    # 2. Replace URLs with a temporary placeholder to prevent them from being broken up
    # Use a unique placeholder that won't naturally occur in text
    placeholder = "__URL_PLACEHOLDER__"
    text_without_urls = re.sub(url_pattern, placeholder, text)

    # 3. Proceed with existing cleaning logic on the text without URLs
    # Lowercase the text
    text_without_urls = text_without_urls.lower()

    # Add spaces around any character that is NOT a letter, number, or whitespace.
    # This isolates punctuation, emojis, and other symbols as separate tokens.
    text_without_urls = re.sub(r'([^a-zA-Z0-9\s])', r' \1 ', text_without_urls)

    # Split text into words (tokenize)
    words = text_without_urls.split()

    # 4. Remove English stop words and short words, and remove the placeholder
    # We keep non-English words because they are not in the stop_words list.
    keywords = [word for word in words if word not in stop_words and len(word) > 1 and word != placeholder.lower()]

    # 5. Add the extracted URLs to the keywords list
    keywords.extend(urls)

    return keywords
#%%
def violations_subreddit():
    # Which subreddits have the highest average violation scores?
    subreddit_violation = df_train.groupby('subreddit')['rule_violation'].mean().sort_values(ascending=False)
    print("\n--- Subreddits with Highest Average Violation Score ---")
    print(subreddit_violation.head(10))
    print("\n--- Subreddits with Lowest Average Violation Score ---")
    print(subreddit_violation.tail(10))
    # See if length correlates with violation score
    print(df_train[['comment_length', 'rule_violation']].corr())
#%%
def prelim_explore():
    print(df_train.info())
    print(df_train.describe())
    # Set pandas to display the full column width
    pd.set_option('display.max_colwidth', None)
    # Look at a few comments with HIGH violation scores
    print("--- High Violation Comments ---")
    print(df_train[df_train['rule_violation'] > 0.8].head(3))
    # Look at a few comments with LOW violation scores
    print("\n--- Low Violation Comments ---")
    print(df_train[df_train['rule_violation'] < 0.2].head(3))
#%%
def build_keywords():
    global significant_words
    # This is the key step: transform the data to one-row-per-word
    # This will create a much larger DataFrame
    word_df = df_train.explode('keywords')
    print(f"Shape before exploding: {df_train.shape}")
    print(f"Shape after exploding: {word_df.shape}")
    # Now, group by each keyword and calculate its stats
    print("\nCalculating word scores... (This may take a minute)")
    word_scores = word_df.groupby('keywords')['rule_violation'].agg(['count', 'mean']).reset_index()
    word_scores.rename(columns={'mean': 'mean_violation'}, inplace=True)
    print("Calculation complete.")
    # --- Filter out rare words to get more meaningful results ---
    # A word needs to appear at least 50 times to be considered.
    # This avoids drawing conclusions from words that only appear a few times.
    min_word_count = 64
    significant_words = word_scores[word_scores['count'] >= min_word_count]
    print(f"Total unique words: {word_scores.shape[0]}")
    print(f"Significant words (>= {min_word_count} occurrences): {significant_words.shape[0]}")
    return significant_words
#%%
def filter_subreddits():
    global top_overall_subreddits
    # --- NEW FILTERING LOGIC FOR SUBREDDITS ---
    min_comments_for_subreddit_analysis = 10  # Minimum comments required for a subreddit to be considered
    print(f"\nFiltering subreddits: requiring at least {min_comments_for_subreddit_analysis} comments and mean violation between 0% and 100%.")
    # Combine into a temporary DataFrame for easy filtering
    subreddit_stats = pd.DataFrame(
        {
            'mean_violation': overall_subreddit_mean_violations, 'comment_count': num_comments_per_subreddit
        }
    )
    # Apply the filtering conditions
    filtered_subreddits = subreddit_stats[(subreddit_stats['comment_count'] >= min_comments_for_subreddit_analysis) & (subreddit_stats['mean_violation'] > 0) &  # Exclude 0% violation
                                          (subreddit_stats['mean_violation'] < 1)  # Exclude 100% violation
                                          ]
    # Get the subreddits from this filtered list, sorted by their mean violation
    return filtered_subreddits.sort_values('mean_violation', ascending=False).head(32).index.tolist()
#%%
def violations_filtered_subreddits(top_overall_subreddits):
    global subreddit, current_subreddit_mean, current_subreddit_count
    # --- Step 5: Identify Top Violators per Subreddit (using the filtered list) ---
    num_subreddits_to_display = 16  # This will now be from the filtered list
    num_keywords_per_subreddit = 16
    print(f"\n--- Top {num_keywords_per_subreddit} Violation-Prone Keywords per Subreddit (for top {num_subreddits_to_display} *filtered* subreddits) ---")
    if not top_overall_subreddits:
        print("No subreddits met the filtering criteria to display.")
    else:
        for subreddit in top_overall_subreddits:
            # Get the overall mean violation and comment count for the current subreddit
            current_subreddit_mean = overall_subreddit_mean_violations.loc[subreddit]
            current_subreddit_count = num_comments_per_subreddit.loc[subreddit]

            print(f"\nSubreddit: r/{subreddit} (Comments: {current_subreddit_count}, Overall Mean Violation: {current_subreddit_mean:.4f})")

            # Filter for the current subreddit and sort by mean_violation
            subreddit_top_keywords = significant_subreddit_keywords[significant_subreddit_keywords['subreddit'] == subreddit].sort_values('mean_violation', ascending=False).head(
                num_keywords_per_subreddit
                )

            if not subreddit_top_keywords.empty:
                print(subreddit_top_keywords[['keywords', 'count', 'mean_violation']].to_string(index=False))
            else:
                print("  No significant keywords found for this subreddit (after keyword occurrence filter).")
#%%
def violations_urls():
    global url_pattern_check, significant_subreddit_urls
    # Define the URL pattern again to filter keywords
    url_pattern_check = re.compile(r'https?://\S+|www\.\S+')
    # Assuming subreddit_keyword_df and url_pattern_check are already defined
    # Filter the exploded DataFrame to only include URLs as keywords
    # (This is the same url_keyword_df from Part 1)
    url_keyword_df = subreddit_keyword_df[subreddit_keyword_df['keywords'].apply(lambda x: isinstance(x, str) and bool(url_pattern_check.match(x)))].copy()
    # Group by subreddit and URL, then aggregate
    print("\nCalculating mean violation for each URL per subreddit...")
    subreddit_url_scores = url_keyword_df.groupby(['subreddit', 'keywords'])['rule_violation'].agg(['count', 'mean']).reset_index()
    subreddit_url_scores.rename(columns={'mean': 'mean_violation', 'keywords': 'url'}, inplace=True)
    # Filter for significance (e.g., URL appearing at least 3 times within a subreddit)
    min_url_occurrences_per_subreddit = 3
    significant_subreddit_urls = subreddit_url_scores[subreddit_url_scores['count'] >= min_url_occurrences_per_subreddit]
    print(f"Found {significant_subreddit_urls.shape[0]} significant (subreddit, URL) pairs (>= {min_url_occurrences_per_subreddit} occurrences).")
#%%
def urls_subredditts():
    global subreddit, current_subreddit_mean, current_subreddit_count
    # --- Display Top URLs per Subreddit ---
    # Reuse the filtered_subreddits and top_overall_subreddits from previous analysis
    # (assuming they are still in your notebook's environment)
    num_subreddits_to_display_urls = 5
    num_urls_per_subreddit = 10
    print(f"\n--- Top {num_urls_per_subreddit} Violation-Prone URLs per Subreddit (for top {num_subreddits_to_display_urls} filtered subreddits) ---")
    if not top_overall_subreddits:  # This list comes from the previous subreddit filtering
        print("No subreddits met the filtering criteria to display URLs.")
    else:
        for subreddit in top_overall_subreddits:
            # Get the overall mean violation and comment count for the current subreddit
            current_subreddit_mean = overall_subreddit_mean_violations.loc[subreddit]
            current_subreddit_count = num_comments_per_subreddit.loc[subreddit]

            print(f"\nSubreddit: r/{subreddit} (Comments: {current_subreddit_count}, Overall Mean Violation: {current_subreddit_mean:.4f})")

            # Filter for the current subreddit and sort by mean_violation
            subreddit_top_urls = significant_subreddit_urls[significant_subreddit_urls['subreddit'] == subreddit].sort_values('mean_violation', ascending=False).head(num_urls_per_subreddit)

            if not subreddit_top_urls.empty:
                print(subreddit_top_urls[['url', 'count', 'mean_violation']].to_string(index=False))
            else:
                print("  No significant URLs found for this subreddit.")
#%%
def subreddit_body_violation():
    global subreddit_keyword_df, significant_subreddit_keywords, overall_subreddit_mean_violations, num_comments_per_subreddit, top_overall_subreddits
    # Ensure full comment text is displayed for later inspection
    pd.set_option('display.max_colwidth', None)
    # --- Step 1: Explode the DataFrame by keywords (from previous code) ---
    print("Exploding DataFrame by keywords...")
    subreddit_keyword_df = df_train.explode('keywords')
    subreddit_keyword_df.dropna(subset=['keywords'], inplace=True)
    print(f"Shape after exploding: {subreddit_keyword_df.shape}")
    # --- Step 2 & 3: Group by subreddit and keyword, then aggregate (from previous code) ---
    print("Grouping by subreddit and keyword to calculate scores...")
    subreddit_keyword_scores = subreddit_keyword_df.groupby(['subreddit', 'keywords'])['rule_violation'].agg(['count', 'mean']).reset_index()
    subreddit_keyword_scores.rename(columns={'mean': 'mean_violation'}, inplace=True)
    print("Calculation complete.")
    # --- Step 4: Filter for significance (from previous code) ---
    min_occurrences_per_subreddit = 10
    significant_subreddit_keywords = subreddit_keyword_scores[subreddit_keyword_scores['count'] >= min_occurrences_per_subreddit]
    print(f"Total unique (subreddit, keyword) pairs: {subreddit_keyword_scores.shape[0]}")
    print(f"Significant (subreddit, keyword) pairs (>= {min_occurrences_per_subreddit} occurrences): {significant_subreddit_keywords.shape[0]}")
    # --- Calculate overall mean violation per subreddit AND number of comments per subreddit ---
    overall_subreddit_mean_violations = df_train.groupby('subreddit')['rule_violation'].mean()
    num_comments_per_subreddit = df_train.groupby('subreddit').size()
    top_overall_subreddits = filter_subreddits()
    violations_filtered_subreddits(top_overall_subreddits)
    violations_urls()
    urls_subredditts()
#%%
def subreddit_body_violation():
    global subreddit_keyword_df, significant_subreddit_keywords, overall_subreddit_mean_violations, num_comments_per_subreddit, top_overall_subreddits
    # Ensure full comment text is displayed for later inspection
    pd.set_option('display.max_colwidth', None)
    # --- Step 1: Explode the DataFrame by keywords (from previous code) ---
    print("Exploding DataFrame by keywords...")
    subreddit_keyword_df = df_train.explode('keywords')
    subreddit_keyword_df.dropna(subset=['keywords'], inplace=True)
    print(f"Shape after exploding: {subreddit_keyword_df.shape}")
    # --- Step 2 & 3: Group by subreddit and keyword, then aggregate (from previous code) ---
    print("Grouping by subreddit and keyword to calculate scores...")
    subreddit_keyword_scores = subreddit_keyword_df.groupby(['subreddit', 'keywords'])['rule_violation'].agg(['count', 'mean']).reset_index()
    subreddit_keyword_scores.rename(columns={'mean': 'mean_violation'}, inplace=True)
    print("Calculation complete.")
    # --- Step 4: Filter for significance (from previous code) ---
    min_occurrences_per_subreddit = 10
    significant_subreddit_keywords = subreddit_keyword_scores[subreddit_keyword_scores['count'] >= min_occurrences_per_subreddit]
    print(f"Total unique (subreddit, keyword) pairs: {subreddit_keyword_scores.shape[0]}")
    print(f"Significant (subreddit, keyword) pairs (>= {min_occurrences_per_subreddit} occurrences): {significant_subreddit_keywords.shape[0]}")
    # --- Calculate overall mean violation per subreddit AND number of comments per subreddit ---
    overall_subreddit_mean_violations = df_train.groupby('subreddit')['rule_violation'].mean()
    num_comments_per_subreddit = df_train.groupby('subreddit').size()
    top_overall_subreddits = filter_subreddits()
    violations_filtered_subreddits(top_overall_subreddits)
    violations_urls()
    urls_subredditts()
#%%
def rule_subredditt():
    # Assuming df_full_train is already loaded
    # df_full_train = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
    print("--- Analyzing Rule-Subreddit Relationships ---")
    # 1. Number of unique rules per subreddit
    print("\n--- Number of Unique Rules per Subreddit ---")
    rules_per_subreddit = df_train.groupby('subreddit')['rule'].nunique().sort_values(ascending=False)
    print("Top 10 subreddits by number of unique rules:")
    print(rules_per_subreddit.head(10))
    print("\nBottom 10 subreddits by number of unique rules:")
    print(rules_per_subreddit.tail(10))
    print(f"\nTotal unique subreddits: {len(rules_per_subreddit)}")
    print(f"Subreddits with only one rule: {len(rules_per_subreddit[rules_per_subreddit == 1])}")
    # 2. Number of unique subreddits per rule
    print("\n--- Number of Unique Subreddits per Rule ---")
    subreddits_per_rule = df_train.groupby('rule')['subreddit'].nunique().sort_values(ascending=False)
    print("Top 10 rules by number of unique subreddits:")
    print(subreddits_per_rule.head(10))
    print("\nBottom 10 rules by number of unique subreddits:")
    print(subreddits_per_rule.tail(10))
    print(f"\nTotal unique rules: {len(subreddits_per_rule)}")
    print(f"Rules appearing in only one subreddit: {len(subreddits_per_rule[subreddits_per_rule == 1])}")
#%%
def rule_violations():
    # Ensure full comment text is displayed for later inspection
    pd.set_option('display.max_colwidth', None)
    # Assuming df_full_train is loaded and 'keywords' column is generated by clean_and_tokenize_with_urls
    # (using df_full_train['body'].apply(clean_and_tokenize_with_urls))
    # --- Step 1: Explode the DataFrame by keywords ---
    # This creates a row for each keyword, duplicating the comment's rule and violation score.
    print("Exploding DataFrame by keywords...")
    rule_keyword_df = df_train.explode('keywords')
    rule_keyword_df.dropna(subset=['keywords'], inplace=True)
    print(f"Shape after exploding: {rule_keyword_df.shape}")
    # --- Step 2 & 3: Group by rule and keyword, then aggregate ---
    print("Grouping by rule and keyword to calculate scores...")
    rule_keyword_scores = rule_keyword_df.groupby(['rule', 'keywords'])['rule_violation'].agg(['count', 'mean']).reset_index()
    rule_keyword_scores.rename(columns={'mean': 'mean_violation'}, inplace=True)
    print("Calculation complete.")
    # --- Step 4: Filter for significance ---
    # A word needs to appear at least 'min_occurrences_per_rule' times within a rule
    # to be considered for analysis. Adjust this threshold as needed.
    min_occurrences_per_rule = 16  # Similar to subreddit, adjust based on data
    significant_rule_keywords = rule_keyword_scores[rule_keyword_scores['count'] >= min_occurrences_per_rule]
    print(f"Total unique (rule, keyword) pairs: {rule_keyword_scores.shape[0]}")
    print(f"Significant (rule, keyword) pairs (>= {min_occurrences_per_rule} occurrences): {significant_rule_keywords.shape[0]}")
    # --- NEW: Calculate overall mean violation per rule AND number of comments per rule ---
    overall_rule_mean_violations = df_train.groupby('rule')['rule_violation'].mean()
    num_comments_per_rule = df_train.groupby('rule').size()
    # --- Step 5: Identify Top Violators per Rule ---
    num_rules_to_display = 2
    num_keywords_per_rule = 16
    # Get the rules with the highest overall average violation scores (for display order)
    # We'll apply similar filtering for rules as we did for subreddits to avoid skewed results
    min_comments_for_rule_analysis = 4  # Minimum comments required for a rule to be considered
    rule_stats = pd.DataFrame(
        {
            'mean_violation': overall_rule_mean_violations, 'comment_count': num_comments_per_rule
        }
    )
    filtered_rules = rule_stats[(rule_stats['comment_count'] >= min_comments_for_rule_analysis) & (rule_stats['mean_violation'] > 0) & (rule_stats['mean_violation'] < 1)]
    top_overall_rules = filtered_rules.sort_values('mean_violation', ascending=False).head(num_rules_to_display).index.tolist()
    print(f"\n--- Top {num_keywords_per_rule} Violation-Prone Keywords per Rule (for top {num_rules_to_display} *filtered* rules) ---")
    if not top_overall_rules:
        print("No rules met the filtering criteria to display.")
    else:
        for rule in top_overall_rules:
            # Get the overall mean violation and comment count for the current rule
            current_rule_mean = overall_rule_mean_violations.loc[rule]
            current_rule_count = num_comments_per_rule.loc[rule]

            print(f"\nRule: {rule} (Comments: {current_rule_count}, Overall Mean Violation: {current_rule_mean:.4f})")

            # Filter for the current rule and sort by mean_violation
            rule_top_keywords = significant_rule_keywords[significant_rule_keywords['rule'] == rule].sort_values('mean_violation', ascending=False).head(num_keywords_per_rule)

            if not rule_top_keywords.empty:
                print(rule_top_keywords[['keywords', 'count', 'mean_violation']].to_string(index=False))
            else:
                print("  No significant keywords found for this rule (after keyword occurrence filter).")
#%%
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value) # For all GPUs
    # For deterministic behavior (can slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
#%%
def create_datasets():
    # --- Tokenizer ---
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_PATH,  use_fast=False)

    # --- Dataset Class ---
    class JigsawDataset(Dataset):
        def __init__(self, dataframe, tokenizer, max_len, is_test=False):
            self.dataframe = dataframe
            self.tokenizer = tokenizer
            self.max_len = max_len
            self.is_test = is_test

            # Create the enriched rule string for both train and test
            # Use .fillna('') to handle potential NaN values in example columns gracefully
            # This is the core of the "enriched rule" strategy
            self.dataframe['enriched_rule'] = (
                "RULE: " + self.dataframe['rule'] + " POS_EX: " + self.dataframe['positive_example_1'].fillna('') + " " + self.dataframe['positive_example_2'].fillna('') + " NEG_EX: " +
                self.dataframe['negative_example_1'].fillna('') + " " + self.dataframe['negative_example_2'].fillna(''))

            # Combine 'body' and 'enriched_rule' for input to the model
            # The tokenizer.sep_token acts as a separator that the model understands
            # This creates a single input string for the transformer
            self.texts = self.dataframe.apply(lambda x: x['body'] + tokenizer.sep_token + x['enriched_rule'], axis=1)

            if not is_test:
                self.labels = self.dataframe['rule_violation'].values  # No else needed for test, as labels are not used for test

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            text = str(self.texts.iloc[idx])

            # Tokenize the text
            encoding = self.tokenizer.encode_plus(
                text, add_special_tokens=True, max_length=self.max_len, return_token_type_ids=False,  # RoBERTa and XLM-R models typically don't use token_type_ids
                padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt', )

            inputs = {
                'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten()
            }

            if not self.is_test:
                # Ensure labels are float32 and have a shape of [1] for BCEWithLogitsLoss
                inputs['labels'] = torch.tensor([self.labels[idx]], dtype=torch.float32)

            return inputs

    # --- Create Datasets and DataLoaders ---
    MAX_LEN = 256  # Max sequence length for DistilBERT. Adjust based on your text length.
    BATCH_SIZE = 16  # Smaller batch size for memory constraints
    # Split training data for local validation (good practice)
    train_df, val_df = train_test_split(df_train, test_size=0.1, random_state=42)
    train_dataset = JigsawDataset(train_df, tokenizer, MAX_LEN)
    val_dataset = JigsawDataset(val_df, tokenizer, MAX_LEN)
    test_dataset = JigsawDataset(df_test, tokenizer, MAX_LEN, is_test=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("Datasets and DataLoaders created.")

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader

#%%
# Path to the training data inside your Kaggle Notebook
trn_file = "/kaggle/input/jigsaw-agile-community-rules/train.csv"
tst_file = "/kaggle/input/jigsaw-agile-community-rules/test.csv"

# Load it into a pandas DataFrame
df_train = pd.read_csv(trn_file)
df_test  = pd.read_csv(tst_file)

# prelim_explore()

# Create a new feature for comment length
df_train['comment_length'] = df_train['body'].str.len()

# violations_subreddit()

# --- Create a 'keywords' column ---
print("Applying URL-aware cleaning function...")
df_train['keywords'] = df_train['body'].apply(clean_and_tokenize_with_urls)
print("Cleaning complete.")

# pd.set_option('display.max_colwidth', 300)
# print(df_train[df_train['body'].str.contains(r'[^a-zA-Z0-9\s]', na=False)][['body', 'keywords']].head())
significant_words = build_keywords()

# --- Words with the HIGHEST violation scores ---
top_violators = significant_words.sort_values('mean_violation', ascending=False)

# print("\n--- Top 20 Words Correlated with HIGH Violation Scores ---")
# print(top_violators.head(20))

# --- Words with the LOWEST violation scores ---
bottom_violators = significant_words.sort_values('mean_violation', ascending=True)

# print("\n--- Top 20 Words Correlated with LOW Violation Scores ---")
# print(bottom_violators.head(20))

pd.set_option('display.max_colwidth', None)

# --- 1. Get the lists of keywords ---
num_keywords_to_search = 64
top_keywords_list = top_violators.head(num_keywords_to_search)['keywords'].tolist()
bottom_keywords_list = bottom_violators.head(num_keywords_to_search)['keywords'].tolist()

# --- 2. Build the regex patterns ---
top_regex = '|'.join([re.escape(word) for word in top_keywords_list])
bottom_regex = '|'.join([re.escape(word) for word in bottom_keywords_list])

print(f"Searching for high-violation keywords like: {top_keywords_list[:5]}")
print(f"Searching for low-violation keywords like: {bottom_keywords_list[:5]}")

# --- 3. Filter the DataFrame to find comments containing these words ---
high_violation_examples = df_train[df_train['body'].str.contains(top_regex, case=False, na=False)]
low_violation_examples = df_train[df_train['body'].str.contains(bottom_regex, case=False, na=False)]

subreddit_body_violation()
# rule_subredditt()
rule_violations()
#%%
# --- The device ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using NVIDIA CUDA GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("NVIDIA CUDA GPU not available, using CPU.")

SEED = 42
set_seed(SEED)

MODEL_PATH = "/kaggle/input/xlm-roberta-base-offline/xlm_roberta_base_offline"

train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = create_datasets()

# --- Load Model ---
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=1)
model.to(device) # Move model to device

# --- Optimizer and Loss Function ---
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
# BCEWithLogitsLoss is good for binary classification when the model outputs logits (raw scores)
loss_fn = torch.nn.BCEWithLogitsLoss()

# --- Training Parameters ---
N_EPOCHS = 8

print(f"Starting training for {N_EPOCHS} epoch(s)...")
for epoch in range(N_EPOCHS):
    model.train() # Set model to training mode
    train_loss = 0
    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device) # Labels will now be [batch_size, 1]

        optimizer.zero_grad() # Clear gradients

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # *** FIX HERE: Keep logits as [batch_size, 1] for BCEWithLogitsLoss ***
        logits = outputs.logits

        loss = loss_fn(logits, labels) # Both logits and labels are [batch_size, 1]
        loss.backward() # Backpropagation
        optimizer.step() # Update weights

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1} - Avg Train Loss: {avg_train_loss:.4f}")

    # --- Validation (Optional, but good for monitoring) ---
    model.eval() # Set model to evaluation mode
    val_preds = []
    val_true = []
    val_loss = 0
    with torch.no_grad(): # Disable gradient calculation for validation
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device) # Labels will be [batch_size, 1]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # *** FIX HERE: Keep logits as [batch_size, 1] ***
            logits = outputs.logits

            loss = loss_fn(logits, labels)
            val_loss += loss.item()

            # *** FIX HERE: Squeeze before sigmoid for prediction ***
            preds = torch.sigmoid(logits.squeeze(-1)).cpu().numpy() # Convert logits to probabilities
            val_preds.extend(preds)
            val_true.extend(labels.squeeze(-1).cpu().numpy()) # Squeeze labels for AUC calculation

    avg_val_loss = val_loss / len(val_loader)
    val_auc = roc_auc_score(val_true, val_preds)
    print(f"Epoch {epoch+1} - Avg Val Loss: {avg_val_loss:.4f} - Val AUC: {val_auc:.4f}")

print("Training complete.")

# --- Make Predictions on Test Data ---
print("Making predictions on test data...")
model.eval() # Set model to evaluation mode
test_predictions = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting on Test Data"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # *** FIX HERE: Squeeze before sigmoid for prediction ***
        logits = outputs.logits

        preds = torch.sigmoid(logits.squeeze(-1)).cpu().numpy() # Convert logits to probabilities
        test_predictions.extend(preds)

print("Predictions generated.")

# --- Create Submission File ---
print("Creating submission.csv file...")
submission_df = pd.DataFrame({
    'row_id': df_test['row_id'],
    'rule_violation': test_predictions
})

submission_df.to_csv('submission.csv', index=False)

print("submission.csv created successfully!")
print(submission_df.head(10))