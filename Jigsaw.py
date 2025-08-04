import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import torch_directml
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

# -----------------------------
# Load and preprocess data
# -----------------------------
# Use Kaggle paths when running on Kaggle

# trn = "/kaggle/input/jigsaw-agile-community-rules/train.csv"
# tst = "/kaggle/input/jigsaw-agile-community-rules/test.csv"
trn = "C:/Users/satra/Downloads/jigsaw-agile-community-rules/train.csv"
tst = "C:/Users/satra/Downloads/jigsaw-agile-community-rules/test.csv"
df_trn = pd.read_csv(trn)
# df_trn = df_trn.sample(frac=.05, random_state=42).reset_index(drop=True)
df_tst = pd.read_csv(tst)
# *** ADD THIS LINE: Prepare df_tst for MultiInputDataset ***
df_tst['text_to_classify'] = df_tst['body'].apply(getText) # Use getText for consistency


def get_device():
    # Try to detect NVIDIA CUDA GPU first
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using NVIDIA CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        return device

    # If no NVIDIA CUDA GPU, try to detect DirectML GPU
    # try:
    #     if torch_directml.is_available():
    #         device = torch_directml.device()
    #         print(f"Using DirectML GPU: {device}")
    #         # Add a small test to ensure it's truly usable
    #         try:
    #             _ = torch.tensor([1], device=device)
    #         except Exception as e:
    #             print(f"Warning: DirectML device found but not usable ({e}). Falling back to CPU.")
    #             return torch.device("cpu")
    #         return device
    #     else:
    #         print("DirectML is NOT available.")
    # except ImportError:
    #     print("torch_directml not installed.")
    # except Exception as e:
    #     print(f"Error checking DirectML: {e}. Falling back to CPU.")

    # If neither GPU is found, fall back to CPU
    device = torch.device("cpu")
    print("No GPU (NVIDIA CUDA or DirectML) found. Using CPU.")
    return device


def fill_empty_examples_pandas(df):
    example_cols = ['positive_example_1', 'positive_example_2', 'negative_example_1', 'negative_example_2']
    for col in example_cols:
        df[col] = df[col].fillna('').astype(str)

    df['positive_example_1'] = df['positive_example_1'].mask(df['positive_example_1'] == '', df['positive_example_2'])
    df['positive_example_2'] = df['positive_example_2'].mask(df['positive_example_2'] == '', df['positive_example_1'])

    df['negative_example_1'] = df['negative_example_1'].mask(df['negative_example_1'] == '', df['negative_example_2'])
    df['negative_example_2'] = df['negative_example_2'].mask(df['negative_example_2'] == '', df['negative_example_1'])

    return df


def getText(value):
    return str(value) if pd.notna(value) else ''


def extract_texts(row):
    return {
        "body": getText(row["body"]),
        "rule": getText(row["rule"]),
        "subreddit": getText(row["subreddit"]),
        "pos1": f"{getText(row['positive_example_1'])}",
        "pos2": f"{getText(row['positive_example_2'])}",
        "neg1": f"{getText(row['negative_example_1'])}",
        "neg2": f"{getText(row['negative_example_2'])}",
    }

df_trn = fill_empty_examples_pandas(df_trn)
df_tst = fill_empty_examples_pandas(df_tst)

df_trn["inputs"] = df_trn.apply(extract_texts, axis=1)
df_tst["inputs"] = df_tst.apply(extract_texts, axis=1) # Apply to test data too

text_feature_cols = [
    'body',
    'rule',
    'subreddit',
    'positive_example_1',
    'positive_example_2',
    'negative_example_1',
    'negative_example_2'
]

print("--- Comprehensive NaN Inspection for All Text Feature Columns ---")

# 1. Count NaNs for each text feature column
print("\n--- NaN Counts per Text Feature Column ---")
print(df_trn[text_feature_cols].isnull().sum())

# 2. Analyze rows with NaNs in 'body' (most critical)
print("\n--- Analysis for 'body' column NaNs ---")
body_nan_rows = df_trn[df_trn['body'].isnull()]
if not body_nan_rows.empty:
    print(f"Number of rows with NaN in 'body': {len(body_nan_rows)}")
    print("Rule violation distribution for rows with NaN in 'body':")
    print(body_nan_rows['rule_violation'].value_counts(normalize=True))
else:
    print("No NaN values found in 'body' column.")

# 3. Analyze rows with NaNs in 'rule'
print("\n--- Analysis for 'rule' column NaNs ---")
rule_nan_rows = df_trn[df_trn['rule'].isnull()]
if not rule_nan_rows.empty:
    print(f"Number of rows with NaN in 'rule': {len(rule_nan_rows)}")
    print("Rule violation distribution for rows with NaN in 'rule':")
    print(rule_nan_rows['rule_violation'].value_counts(normalize=True))
else:
    print("No NaN values found in 'rule' column.")

# 4. Analyze rows with NaNs in 'subreddit'
print("\n--- Analysis for 'subreddit' column NaNs ---")
subreddit_nan_rows = df_trn[df_trn['subreddit'].isnull()]
if not subreddit_nan_rows.empty:
    print(f"Number of rows with NaN in 'subreddit': {len(subreddit_nan_rows)}")
    print("Rule violation distribution for rows with NaN in 'subreddit':")
    print(subreddit_nan_rows['rule_violation'].value_counts(normalize=True))
else:
    print("No NaN values found in 'subreddit' column.")

# 5. Analyze rows where ANY of the example columns are NaN
print("\n--- Analysis for Example Columns NaNs ---")
example_only_cols = [col for col in text_feature_cols if 'example' in col]
df_any_example_nan = df_trn[df_trn[example_only_cols].isnull().any(axis=1)]
if not df_any_example_nan.empty:
    print(f"Number of rows with NaN in ANY example column: {len(df_any_example_nan)}")
    print("Rule violation distribution for rows with NaN in ANY example column:")
    print(df_any_example_nan['rule_violation'].value_counts(normalize=True))
else:
    print("No NaN values found in any example column.")

# 6. Overall rule_violation distribution (for comparison)
print(f"\n--- Overall rule_violation distribution: ---")
print(df_trn['rule_violation'].value_counts(normalize=True))


N_EPOCHS = 8
k_folds = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# -----------------------------
# Dataset
# -----------------------------
class MultiInputDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128, is_test=False): # Renamed df_trn to df for generality
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text_inputs = row["inputs"]
        item = {}
        for field in ["text_to_classify", "rule", "subreddit"]:
            encoded = self.tokenizer(
                text_inputs[field],
                truncation=True,
                padding='max_length',
                max_length=self.max_len,
                return_tensors="pt"
            )

            for key in encoded:
                item[f"{field}_{key}"] = encoded[key].squeeze(0)
        if not self.is_test:
          item["label"] = torch.tensor(row["rule_violation"], dtype=torch.float32)
        return item

# -----------------------------
# Model
# -----------------------------
class MultiInputBERT(nn.Module):
    def __init__(self, model_name='xlm-roberta-base'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(768 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # Output a single logit
        )

    def forward(self, inputs):
        cls_outputs = []
        for field in ["text_to_classify", "rule", "subreddit"]:
            out = self.bert(
                input_ids=inputs[f"{field}_input_ids"],
                attention_mask=inputs[f"{field}_attention_mask"]
            )
            cls_outputs.append(out.last_hidden_state[:, 0])  # CLS token
        x = torch.cat(cls_outputs, dim=1)
        x = self.dropout(x)
        return self.classifier(x) # Return raw logits

# -----------------------------
# Training and Evaluation
# -----------------------------
device = get_device()
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")


oof_preds = np.zeros(len(df_trn))
test_preds_folds = [] # This is correct

test_loader = DataLoader(MultiInputDataset(df_tst, tokenizer), batch_size=4, shuffle=False)

for fold, (train_idx, val_idx) in enumerate(skf.split(df_trn, df_trn["rule_violation"])):
    print(f"\n----- Fold {fold+1} -----")
    train_df = df_trn.iloc[train_idx].reset_index(drop=True)

    # Create original train and validation DataFrames for this fold
    # These are the original body, rules, subreddits, and examples
    fold_train_df_orig = df_trn.iloc[train_idx_orig].reset_index(drop=True)
    fold_val_df_orig = df_trn.iloc[val_idx_orig].reset_index(drop=True)

    # 2. EXPAND the TRAINING data for this fold
    expanded_train_data = []
    for idx, row in fold_train_df_orig.iterrows():
        rule_text = getText(row['rule'])
        subreddit_text = getText(row['subreddit'])
        # Add original body as a training sample
        expanded_train_data.append({
            'text_to_classify': getText(row['body']),
            'rule': rule_text,
            'subreddit': subreddit_text,
            'rule_violation': row['rule_violation']
        })
        # Add positive examples
        expanded_train_data.append({
            'text_to_classify': getText(row['positive_example_1']),
            'rule': rule_text,
            'subreddit': subreddit_text,
            'rule_violation': 1.0
        })
        expanded_train_data.append({
            'text_to_classify': getText(row['positive_example_2']),
            'rule': rule_text,
            'subreddit': subreddit_text,
            'rule_violation': 1.0
        })
        # Add negative examples
        expanded_train_data.append({
            'text_to_classify': getText(row['negative_example_1']),
            'rule': rule_text,
            'subreddit': subreddit_text,
            'rule_violation': 0.0
        })
        expanded_train_data.append({
            'text_to_classify': getText(row['negative_example_2']),
            'rule': rule_text,
            'subreddit': subreddit_text,
            'rule_violation': 0.0
        })

    # Create the expanded training DataFrame for this fold
    fold_train_df_expanded = pd.DataFrame(expanded_train_data)
    fold_train_df_expanded = fold_train_df_expanded[fold_train_df_expanded['text_to_classify'] != ''].reset_index(drop=True)

    # 3. Prepare the VALIDATION data for this fold (using original body)
    # Map 'body' to 'text_to_classify' for the validation set
    fold_val_df_for_model = fold_val_df_orig.copy()
    fold_val_df_for_model['text_to_classify'] = fold_val_df_for_model['body']

    # 4. Create Datasets and DataLoaders
    train_dataset = MultiInputDataset(fold_train_df_expanded, tokenizer) # Train on expanded data
    val_dataset = MultiInputDataset(fold_val_df_for_model, tokenizer, is_test=True) # Validate on original body

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8) # Use a consistent batch size

    test_loader = DataLoader(MultiInputDataset(df_tst, tokenizer, is_test=True), batch_size=16, shuffle=False)


    model = MultiInputBERT().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    num_training_steps_per_fold = len(train_loader) * N_EPOCHS
    num_warmup_steps_per_fold = int(num_training_steps_per_fold * 0.05)

    # Initialize the scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps_per_fold,
        num_training_steps=num_training_steps_per_fold
    )

    # Training Loop for this fold
    best_auc = -1.0 # Track best AUC for this fold
    best_model_state = None # To save the best model for this fold

    for epoch in range(N_EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            logits = outputs.squeeze(-1)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

        # Eval
        model.eval()
        preds_raw, labels_all = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}"):
                inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
                labels = batch["label"].to(device)
                outputs = model(inputs)
                logits = outputs.squeeze(-1) # Squeeze to [batch_size]

                probs = torch.sigmoid(logits).detach().cpu().tolist()
                preds_raw.extend(probs)
                labels_all.extend(labels.cpu().tolist())

            # Hard labels (for classification report, optional)
            preds = [int(p > 0.5) for p in preds_raw]

            # Print metrics
            print(classification_report(labels_all, preds, digits=3, zero_division=0))

            curr_auc = roc_auc_score(labels_all, preds_raw)
            print(f"AUC Score: {curr_auc:.4f}")

            # Save the best model for this fold based on validation AUC
            if curr_auc > best_auc:
                best_auc = curr_auc
                best_model_state = model.state_dict() # Save model weights
                print(f"  -> New best Val AUC for Fold {fold+1}: {best_auc:.4f}")

    # 6. Load best model state for this fold
    model.load_state_dict(best_model_state) # Use best_model_state
    print(f"Fold {fold+1} Best Val AUC: {best_auc:.4f}")

    # Make OOF predictions for this fold's validation set
    model.eval()
    fold_val_preds_list = []
    fold_val_true_list = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Fold {fold+1} OOF Prediction"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            labels = batch["label"].to(device)
            outputs = model(inputs)
            logits = outputs.squeeze(-1) # Squeeze to [batch_size]
            probs = torch.sigmoid(logits).detach().cpu().tolist()
            fold_val_preds_list.extend(probs)
           # Get true labels from the original validation DataFrame
           # This assumes val_loader iterates in the same order as fold_val_df_for_model
           # which it should if shuffle=False
           fold_val_true_list.extend(
             fold_val_df_orig['rule_violation'].iloc[batch.get('idx', 
             range(len(batch[text_to_classify_input_ids'])))].tolist()
        ) # More robust way to get labels

   # Sanity check: Calculate AUC for this fold's OOF predictions
   oof_fold_auc_check = roc_auc_score(fold_val_true_list, fold_val_preds_list)
   print(f"Fold {fold+1} OOF AUC Check: {oof_fold_auc_check:.4f} (This is the true validation AUC for this fold)")

   # *** CRITICAL FIX: Assign predictions to the correct indices in the global oof_preds array ***
   oof_preds[val_idx_orig] = np.array(fold_val_preds_list) # Use val_idx_orig from kf.split
    # Make predictions on the TEST set using this fold's best model
    test_fold_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Fold {fold+1} Test Prediction"):
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(inputs)
            logits = outputs.squeeze(-1) # Squeeze to [batch_size]
            probs = torch.sigmoid(logits).detach().cpu().tolist()
            test_fold_preds.extend(probs)

    test_preds_folds.append(test_fold_preds) # Store test predictions from this fold


# -----------------------------
# Final Calculation and Submission
# -----------------------------
overall_oof_auc = roc_auc_score(df_trn['rule_violation'], oof_preds)
print(f"\n--- Overall {k_folds}-Fold OOF AUC: {overall_oof_auc:.4f} ---")

# Average test predictions across all folds
final_test_predictions = np.mean(test_preds_folds, axis=0)

# Create final submission file
submission = pd.DataFrame({
    "row_id": df_tst["row_id"],
    "rule_violation": final_test_predictions
})
submission.to_csv("submission.csv", index=False) # Save with a distinct name
print("K-Fold multi-input submission.csv created successfully!")
print(submission.head(10))








