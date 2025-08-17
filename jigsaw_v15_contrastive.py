# !pip install pytorch_metric_learning
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# import torch_directml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup, pipeline

# -----------------------------
# Paths and Hyperparameters
# -----------------------------
MODEL_PATH = "C:/Users/satra/Downloads/xlm_roberta_base_offline"
TRAIN_PATH = "C:/Users/satra/Downloads/jigsaw-agile-community-rules/train.csv"
TEST_PATH = "C:/Users/satra/Downloads/jigsaw-agile-community-rules/test.csv"

N_EPOCHS = 2
k_folds = 5
BATCH_SIZE = 8
MAX_LEN = 128

# -----------------------------
# Load and preprocess data
# -----------------------------
df_trn = pd.read_csv(TRAIN_PATH)
df_trn = df_trn.sample(frac=0.01, random_state=42).reset_index(drop=True)
df_tst = pd.read_csv(TEST_PATH)


def get_device():
    # Try to detect NVIDIA CUDA GPU first
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using NVIDIA CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        return device

    # If no NVIDIA CUDA GPU, try to detect DirectML GPU
    try:
        # if torch_directml.is_available():
        #     device = torch_directml.device()
        #     print(f"Using DirectML GPU: {device}")
        #     # Add a small test to ensure it's truly usable
        #     try:
        #         _ = torch.tensor([1], device=device)
        #     except Exception as e:
        #         print(f"Warning: DirectML device found but not usable ({e}). Falling back to CPU.")
        #         return torch.device("cpu")
        #     return device
        # else:
        print("DirectML is NOT available.")
    except ImportError:
        print("torch_directml not installed.")
    except Exception as e:
        print(f"Error checking DirectML: {e}. Falling back to CPU.")

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


def extract_texts(row):
  return {
    "body": row["body"], "rule": row["rule"], "subreddit": row["subreddit"], "pos1": row['positive_example_1'], "pos2": row['positive_example_2'], "neg1": row['negative_example_1'],
    "neg2": row['negative_example_2'],
  }


df_trn = fill_empty_examples_pandas(df_trn)
df_tst = fill_empty_examples_pandas(df_tst)
df_trn["inputs"] = df_trn.apply(extract_texts, axis=1)
df_tst["inputs"] = df_tst.apply(extract_texts, axis=1)


# Load original training data (assuming it's already in df_trn with the "inputs" column ready)
# If not, make sure df_trn["inputs"] is populated as before

# Load translation pipelines
en_to_de = pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de")
de_to_en = pipeline("translation_de_to_en", model="Helsinki-NLP/opus-mt-de-en")

def back_translate(text, intermediate_lang_fn_1, intermediate_lang_fn_2):
    try:
        translated = intermediate_lang_fn_1(text)[0]['translation_text']
        back_translated = intermediate_lang_fn_2(translated)[0]['translation_text']
        return back_translated
    except Exception as e:
        print("Translation error:", e)
        return text  # Fallback to original if something goes wrong


# 1. Set random seed for reproducibility
random.seed(42)

# 2. Clone subsets for augmentation
df_body_aug = df_trn.sample(frac=0.10, random_state=42).copy()
df_rule_aug = df_trn.sample(frac=0.10, random_state=43).copy()
df_ex_aug = df_trn.sample(frac=0.10, random_state=44).copy()

# 3. Back-translate 'body' only
print("Augmenting body fields...")
for i, row in tqdm(df_body_aug.iterrows(), total=len(df_body_aug)):
    row["inputs"]["body"] = back_translate(row["inputs"]["body"], en_to_de, de_to_en)

# 4. Back-translate 'rule' only
print("Augmenting rule fields...")
for i, row in tqdm(df_rule_aug.iterrows(), total=len(df_rule_aug)):
    row["inputs"]["rule"] = back_translate(row["inputs"]["rule"], en_to_de, de_to_en)

# 5. Back-translate random example fields
example_fields = ["pos1", "pos2", "neg1", "neg2"]
print("Augmenting example fields...")
for i, row in tqdm(df_ex_aug.iterrows(), total=len(df_ex_aug)):
    fields_to_augment = random.sample(example_fields, k=random.randint(1, 4))
    for field in fields_to_augment:
        row["inputs"][field] = back_translate(row["inputs"][field], en_to_de, de_to_en)

# 6. Combine original and augmented
df_trn_augmented = pd.concat([df_trn, df_body_aug, df_rule_aug, df_ex_aug]).reset_index(drop=True)
print(f"Original samples: {len(df_trn)}, After augmentation: {len(df_trn_augmented)}")


# -----------------------------
# Dataset
# -----------------------------
class MultiInputDataset(Dataset):
  def __init__(self, df, tokenizer, max_len=MAX_LEN, is_test=False):
    self.df = df
    self.tokenizer = tokenizer
    self.max_len = max_len
    self.is_test = is_test

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    item = {}
    for field in ["body", "rule", "subreddit", "pos1", "pos2", "neg1", "neg2"]:
      encoded = self.tokenizer(row["inputs"][field], truncation=True, padding='max_length', max_length=self.max_len, return_tensors="pt")
      for key in encoded:
        item[f"{field}_{key}"] = encoded[key].squeeze(0)
    if not self.is_test:
      item["label"] = torch.tensor(row["rule_violation"], dtype=torch.float32)
    return item


# -----------------------------
# Contrastive Loss
# -----------------------------
class SupConLoss(nn.Module):
  def __init__(self, temperature=0.07):
    super().__init__()
    self.temperature = temperature

  def forward(self, features, labels):
    device = features.device
    features = nn.functional.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T) / self.temperature

    logits_mask = torch.eye(features.size(0), device=device).bool()
    similarity_matrix.masked_fill_(logits_mask, -9e15)

    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    mask.fill_diagonal_(0)

    exp_sim = torch.exp(similarity_matrix)
    denom = exp_sim.sum(dim=1, keepdim=True)

    log_prob = similarity_matrix - torch.log(denom + 1e-12)
    loss = - (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)
    return loss.mean()


# -----------------------------
# Model
# -----------------------------
class MultiInputBERT(nn.Module):
  def __init__(self):
    super().__init__()
    self.bert = AutoModel.from_pretrained(MODEL_PATH)
    self.dropout = nn.Dropout(0.3)
    self.classifier = nn.Sequential(
      nn.Linear(768 * 7, 256), nn.ReLU(), nn.Linear(256, 1)
    )

  def forward(self, inputs, return_cls_only=False):
    cls_outputs = []
    for field in ["body", "rule", "subreddit", "pos1", "pos2", "neg1", "neg2"]:
      out = self.bert(input_ids=inputs[f"{field}_input_ids"], attention_mask=inputs[f"{field}_attention_mask"])
      cls_outputs.append(out.last_hidden_state[:, 0])

    x = torch.cat(cls_outputs, dim=1)
    if return_cls_only:
      return x
    x = self.dropout(x)
    return self.classifier(x)


# -----------------------------
# Training
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
oof_preds = np.zeros(len(df_trn))
test_preds_folds = []

for fold, (train_idx, val_idx) in enumerate(skf.split(df_trn, df_trn['rule_violation'])):
  print(f"\n--- Fold {fold + 1} ---")
  train_df = df_trn.iloc[train_idx].reset_index(drop=True)
  val_df = df_trn.iloc[val_idx].reset_index(drop=True)

  train_loader = DataLoader(MultiInputDataset(train_df, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
  val_loader = DataLoader(MultiInputDataset(val_df, tokenizer), batch_size=BATCH_SIZE)
  test_loader = DataLoader(MultiInputDataset(df_tst, tokenizer, is_test=True), batch_size=BATCH_SIZE)

  model = MultiInputBERT().to(device)
  optimizer = AdamW(model.parameters(), lr=5e-6)
  scheduler = get_linear_schedule_with_warmup(optimizer, int(len(train_loader) * N_EPOCHS * 0.05), len(train_loader) * N_EPOCHS)

  contrastive_loss_fn = SupConLoss()
  bce_loss_fn = nn.BCEWithLogitsLoss()

  best_auc = -1
  best_state = None

  best_logits = None
  for epoch in range(N_EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
      inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
      labels = batch["label"].to(device)

      optimizer.zero_grad()
      cls_feats = model(inputs, return_cls_only=True)
      logits = model(inputs).squeeze(-1)

      loss_bce = bce_loss_fn(logits, labels)
      loss_con = contrastive_loss_fn(cls_feats, labels)
      loss = 0.7 * loss_bce + 0.3 * loss_con

      loss.backward()
      optimizer.step()
      scheduler.step()
      total_loss += loss.item()
    print(f"Loss: {total_loss / len(train_loader):.4f}")

    # Validation
    val_logits, val_labels = [], []
    model.eval()
    with torch.no_grad():
      for batch in val_loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
        labels = batch["label"].to(device)
        logits = model(inputs).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        val_logits.extend(probs)
        val_labels.extend(labels.cpu().numpy())

    val_auc = roc_auc_score(val_labels, val_logits)
    print(f"Val AUC: {val_auc:.4f}")

    if val_auc > best_auc:
      best_auc = val_auc
      best_state = model.state_dict()
      best_logits = val_logits

  model.load_state_dict(best_state)
  print(f"Fold {fold + 1} Best AUC: {best_auc:.4f}")
  oof_preds[val_idx] = np.array(best_logits)

  # Test prediction
  model.eval()
  test_probs = []
  with torch.no_grad():
    for batch in test_loader:
      inputs = {k: v.to(device) for k, v in batch.items()}
      logits = model(inputs).squeeze(-1)
      probs = torch.sigmoid(logits).cpu().numpy()
      test_probs.extend(probs)
  test_preds_folds.append(test_probs)

# -----------------------------
# Final output
# -----------------------------
overall_auc = roc_auc_score(df_trn['rule_violation'], oof_preds)
print(f"\nOverall OOF AUC: {overall_auc:.4f}")

submission = pd.DataFrame(
  {
    "row_id": df_tst["row_id"], "rule_violation": np.mean(test_preds_folds, axis=0)
  }
)
submission.to_csv("submission.csv", index=False)
print("submission.csv created.")
