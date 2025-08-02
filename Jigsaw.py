import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# -----------------------------
# Load and preprocess data
# -----------------------------
trn = "C:/Users/satra/Downloads/jigsaw-agile-community-rules/train.csv"
tst = "C:/Users/satra/Downloads/jigsaw-agile-community-rules/test.csv"
df_trn = pd.read_csv(trn).dropna()

df_tst = pd.read_csv(tst).dropna()

def extract_texts(row):
    return {
        "body": row["body"],
        "rule": row["rule"],
        "pos": f"{row['positive_example_1']} {row['positive_example_2']}",
        "neg": f"{row['negative_example_1']} {row['negative_example_1']}",
    }

df_trn["inputs"] = df_trn.apply(extract_texts, axis=1)
# train_df, val_df = train_test_split(df_trn, test_size=0.2, random_state=42, stratify=df_trn["rule_violation"])

k_folds = 4
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# -----------------------------
# Dataset
# -----------------------------
class MultiInputDataset(Dataset):
    def __init__(self, df_trn, tokenizer, max_len=128):
        self.df_trn = df_trn
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df_trn)

    def __getitem__(self, idx):
        row = self.df_trn.iloc[idx]
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
        item["label"] = torch.tensor(row["rule_violation"], dtype=torch.long)
        return item

# -----------------------------
# Model
# -----------------------------
class MultiInputBERT(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(768 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
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
        return self.classifier(x)

# -----------------------------
# Training and Evaluation
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# train_dataset = MultiInputDataset(train_df, tokenizer)
# val_dataset = MultiInputDataset(val_df, tokenizer)
# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=8)
#
# model = MultiInputBERT().to(device)
# optimizer = AdamW(model.parameters(), lr=1e-5)
# criterion = nn.CrossEntropyLoss()
#
# for epoch in range(4):
#     model.train()
#     total_loss = 0
#     for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
#         inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
#         labels = batch["label"].to(device)
#
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")
#
#     # Eval
#     model.eval()
#     preds_raw, labels_all = [], []
#     with torch.no_grad():
#         for batch in val_loader:
#             inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
#             labels = batch["label"].to(device)
#             outputs = model(inputs)
#             try:
#               logits = outputs.logits
#             except AttributeError:
#               # print("Falling back to raw tensor output (custom model)")
#               logits = outputs
#
#             probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()
#             preds_raw += probs if isinstance(probs, list) else [probs]
#             labels_all += labels.cpu().tolist()
#
#         # Hard labels (if you want classification metrics)
#         preds = [int(p > 0.5) for p in preds_raw]
#
#         # Print metrics
#         print(classification_report(labels_all, preds, digits=3))
#         print(f"AUC Score: {roc_auc_score(labels_all, preds_raw):.4f}")

for fold, (train_idx, val_idx) in enumerate(skf.split(df_trn, df_trn["rule_violation"])):
    print(f"\n----- Fold {fold+1} -----")
    train_df = df_trn.iloc[train_idx].reset_index(drop=True)
    val_df = df_trn.iloc[val_idx].reset_index(drop=True)

    train_dataset = MultiInputDataset(train_df, tokenizer)
    val_dataset = MultiInputDataset(val_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    model = MultiInputBERT().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(4):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

        # Eval
        model.eval()
        preds_raw, labels_all = [], []
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
                labels = batch["label"].to(device)
                outputs = model(inputs)
                try:
                  logits = outputs.logits
                except AttributeError:
                  # print("Falling back to raw tensor output (custom model)")
                  logits = outputs

                probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()
                preds_raw += probs if isinstance(probs, list) else [probs]
                labels_all += labels.cpu().tolist()

            # Hard labels (if you want classification metrics)
            preds = [int(p > 0.5) for p in preds_raw]

            # Print metrics
            print(classification_report(labels_all, preds, digits=3))
            score = roc_auc_score(labels_all, preds_raw)
            print(f"AUC Score: {score :.4f}")


# -----------------------------
# Testing and submission
# -----------------------------
df_tst["inputs"] = df_tst.apply(extract_texts, axis=1)


class TestDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
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
        return item

test_dataset = TestDataset(df_tst, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model.eval()
predictions = []

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(batch)
        try:
            logits = outputs.logits
        except AttributeError:
            # print("Falling back to raw tensor output (custom model)")
            logits = outputs

        probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
        predictions.extend(probs)

submission = pd.DataFrame({
    "row_id": df_tst["row_id"],
    "rule_violation": [p[1] for p in predictions]
})
submission.to_csv("submission.csv", index=False)
print(submission.head(10))