import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
# Use AutoTokenizer for flexibility, and specify the model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertModel # Keep BertModel for
MultiInputBERT

# -----------------------------
# Load and preprocess data
# -----------------------------
# Use Kaggle paths when running on Kaggle
trn = "/kaggle/input/jigsaw-agile-community-rules/train.csv"
tst = "/kaggle/input/jigsaw-agile-community-rules/test.csv"

# Load data. Avoid dropna() on the whole df if you plan to fillna('') for examples.
df_trn = pd.read_csv(trn)
df_tst = pd.read_csv(tst)

# Sample for debugging (comment out for full run)
# df_trn = df_trn.sample(frac=.05, random_state=42).reset_index(drop=True)

def extract_texts(row):
    # *** FIX: Correctly include negative_example_2 ***
    # *** Consider adding .fillna('') here if you don't dropna() on the whole df ***
    return {
        "body": str(row["body"]), # Ensure string
        "rule": str(row["rule"]), # Ensure string
        "pos": f"{str(row['positive_example_1']).fillna('')} {str(row['positive_example_2']).fillna('')}",
        "neg": f"{str(row['negative_example_1']).fillna('')} {str(row['negative_example_2']).fillna('')}",
    }

df_trn["inputs"] = df_trn.apply(extract_texts, axis=1)
df_tst["inputs"] = df_tst.apply(extract_texts, axis=1) # Apply to test data too

k_folds = 5 # Changed to 5 for consistency with previous discussion
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# -----------------------------
# Dataset
# -----------------------------
class MultiInputDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128): # Renamed df_trn to df for generality
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text_inputs = row["inputs"]
        item = {}
        for field in ["body", "rule", "pos", "neg"]:
            encoded = self.tokenizer(
                text_inputs[field],
                truncation=True,
                padding='max_length',
                max_length=self.max_len,
                return_tensors="pt"
            )

            for key in encoded:
                item[f"{field}_{key}"] = encoded[key].squeeze(0)
        # *** FIX: Label dtype to float32 for BCEWithLogitsLoss ***
        item["label"] = torch.tensor(row["rule_violation"], dtype=torch.float32)
        return item

# -----------------------------
# Model
# -----------------------------
class MultiInputBERT(nn.Module):
    # *** FIX: Change model_name to xlm-roberta-base ***
    def __init__(self, model_name='xlm-roberta-base'):
        super().__init__()
        # Use AutoModel for flexibility, or XLMRobertaModel if explicit
        self.bert = BertModel.from_pretrained(model_name) # BertModel is fine for xlm-roberta-base
        self.dropout = nn.Dropout(0.3)
        # *** FIX: Change final layer output to 1 for BCEWithLogitsLoss ***
        self.classifier = nn.Sequential(
            nn.Linear(768 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # Output a single logit
        )

    def forward(self, inputs):
        cls_outputs = []
        for field in ["body", "rule", "pos", "neg"]:
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# *** FIX: Use AutoTokenizer and xlm-roberta-base ***
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

# *** FIX: Initialize oof_preds as a NumPy array ***
oof_preds = np.zeros(len(df_trn))
test_preds_folds = [] # This is correct

for fold, (train_idx, val_idx) in enumerate(skf.split(df_trn, df_trn["rule_violation"])):
    print(f"\n----- Fold {fold+1} -----")
    train_df = df_trn.iloc[train_idx].reset_index(drop=True)
    val_df = df_trn.iloc[val_idx].reset_index(drop=True)

    train_dataset = MultiInputDataset(train_df, tokenizer)
    val_dataset = MultiInputDataset(val_df, tokenizer)
    # TestDataset is not needed as a separate class, MultiInputDataset can handle it
    # test_dataset = TestDataset(df_tst, tokenizer) # REMOVE THIS LINE

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    # Create test_loader outside the loop if you want to predict on full test set once
    # Or create it here if you want to predict for each fold's model
    test_loader = DataLoader(MultiInputDataset(df_tst, tokenizer, is_test=True), batch_size=16, shuffle=False)


    model = MultiInputBERT().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    # *** FIX: Change criterion to BCEWithLogitsLoss ***
    criterion = nn.BCEWithLogitsLoss()

    # Training Loop for this fold
    best_auc = -1.0 # Track best AUC for this fold
    best_model_state = None # To save the best model for this fold

    for epoch in range(4): # Use 4 epochs as a starting point for this model
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            # *** FIX: Logits are already raw outputs from classifier ***
            logits = outputs.squeeze(-1) # Squeeze to [batch_size] for BCEWithLogitsLoss

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
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
                # *** FIX: Logits are already raw outputs from classifier ***
                logits = outputs.squeeze(-1) # Squeeze to [batch_size]

                # *** FIX: Use sigmoid for probabilities from BCEWithLogitsLoss ***
                probs = torch.sigmoid(logits).detach().cpu().tolist()
                preds_raw.extend(probs) # Use extend directly
                labels_all.extend(labels.cpu().tolist()) # Use extend directly

            # Hard labels (for classification report, optional)
            preds = [int(p > 0.5) for p in preds_raw]

            # Print metrics
            print(classification_report(labels_all, preds, digits=3))

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
            fold_val_true_list.extend(labels.cpu().tolist())

    oof_fold_auc_check = roc_auc_score(fold_val_true_list, fold_val_preds_list)
    print(f"Fold {fold+1} OOF AUC Check: {oof_fold_auc_check:.4f} (Must match Best Val AUC)")

    # *** CRITICAL FIX: Assign predictions to the correct indices in the global oof_preds array ***
    oof_preds[val_idx] = np.array(fold_val_preds_list) # Use val_idx from kf.split

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

# 10. Average test predictions across all folds
final_test_predictions = np.mean(test_preds_folds, axis=0)

# 11. Create final submission file
submission = pd.DataFrame({
    "row_id": df_tst["row_id"],
    "rule_violation": final_test_predictions
})
submission.to_csv("submission_kfold_multi_input.csv", index=False) # Save with a distinct name
print("K-Fold multi-input submission_kfold_multi_input.csv created successfully!")
print(submission.head(10))
