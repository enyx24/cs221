import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
import pandas as pd
from utils.preprocess import preprocess_text
from torch.nn import CrossEntropyLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model_name = "vinai/phobert-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    use_safetensors=True
).to(device)

EPOCHS = 5
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
MAX_LEN = 256
BEST_MODEL_PATH = "models/best_phobert.pt"

class SentimentDataset(Dataset):
    def __init__(self, df):
        self.texts = [preprocess_text(t) for t in df['text']]
        self.labels = df['new_label'].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/val.csv")
test_df = pd.read_csv("data/test.csv")

train_loader = DataLoader(SentimentDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(SentimentDataset(val_df), batch_size=BATCH_SIZE)
test_loader = DataLoader(SentimentDataset(test_df), batch_size=BATCH_SIZE)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

best_f1 = 0.0
weights = torch.tensor([1.0, 2.0]).to(device)
loss_fn = CrossEntropyLoss(weight=weights)
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc="Training")

    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        loss = loss_fn(logits, batch["labels"]) 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"Avg Train Loss: {avg_loss:.4f}")

    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            val_labels.extend(batch['labels'].cpu().numpy())

    val_f1 = f1_score(val_labels, val_preds, average='macro')
    print(f"Val F1: {val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print("Best model saved.")

model.load_state_dict(torch.load(BEST_MODEL_PATH))
model.eval()

print("\n\nTesting:")
test_preds, test_labels = [], []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        test_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        test_labels.extend(batch['labels'].cpu().numpy())
print(classification_report(test_labels, test_preds, digits=4))
