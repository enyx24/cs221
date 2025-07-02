import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.preprocess import preprocess_text

# Cấu hình
MODEL_NAME = "vinai/phobert-base-v2"
MODEL_PATH = "models/best_phobert_oversampling.pt"
MAX_LEN = 256

# Load tokenizer & model đã fine-tune
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    output_hidden_states=True  # << enable inspect
)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# Nhập văn bản từ người dùng
text = input("Nhập câu cần phân tích: ")
text = preprocess_text(text)

# Tokenize
encoding = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding="max_length",
    max_length=MAX_LEN
)

# Forward
with torch.no_grad():
    outputs = model(**encoding)

# ---- Phần phân loại ----
logits = outputs.logits
pred_class = torch.argmax(logits, dim=1).item()
label_map = {0: "Tiêu cực", 1: "Tích cực"}
print(f"\nDự đoán: {label_map[pred_class]} (label: {pred_class})")

# ---- Inspect hidden states ----
hidden_states = outputs.hidden_states  # Tuple of 13 (embedding + 12 layers)
cls_vector = hidden_states[-1][0][0]   # Lấy [CLS] ở layer cuối (batch 0, token 0)

print(f"\nVector [CLS] ở layer cuối (length {cls_vector.shape[0]}):")
print(cls_vector)

# Ví dụ: in embedding của token thứ 3 tại layer 5
token_index = 3
layer_index = 5
token_vec = hidden_states[layer_index][0][token_index]
print(f"\nToken {token_index} ở layer {layer_index}:")
print(token_vec)
