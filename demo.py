import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.preprocess import preprocess_text

# Cấu hình
MODEL_NAME = "vinai/phobert-base-v2"
MODEL_PATH = "models/best_phobert_oversampling.pt"
MAX_LEN = 256

# Thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Dự đoán từ văn bản người dùng nhập
def predict_sentiment(text):
    text = preprocess_text(text)
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_label].item()

    return pred_label, confidence

# Giao diện console
label_map = {0: "Tích cực", 1: "Tiêu cực"}

print("Nhập nội dung để phân tích cảm xúc (gõ 'exit' để thoát):")
while True:
    user_input = input(">> ")
    if user_input.lower() == "exit":
        break

    label, score = predict_sentiment(user_input)
    print(f"Dự đoán: {label_map[label]} (độ tin cậy: {score:.4f})")
