import re
import emoji
import unidecode
from underthesea import word_tokenize

# ==== 1. Loại bỏ emoji ====
def remove_emoji(text):
    return emoji.replace_emoji(text, replace='')

# ==== 2. Chuẩn hóa unicode & bỏ dấu ====
def normalize_unicode(text):
    return unidecode.unidecode(text)

# ==== 3. Loại bỏ ký tự đặc biệt & số ====
def remove_punctuation_and_number(text):
    return re.sub(r'[^\w\s]', ' ', re.sub(r'\d+', '', text))

# ==== 4. Tách từ có dấu tiếng Việt (word segmentation) ====
def tokenize_vietnamese(text):
    return word_tokenize(text, format="text")

# ==== 5. Loại bỏ stopword (gợi ý: bạn có thể tuỳ biến danh sách stopword tiếng Việt) ====
def remove_stopwords(text, stopwords):
    words = text.split()
    return ' '.join([w for w in words if w not in stopwords])

# ==== 6. Chuẩn hóa khoảng trắng ====
def normalize_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()

# ==== 7. Tổng hợp pipeline xử lý ====
def preprocess_text(text, 
    stopwords_set=None, 
    lowercase_flag=True, 
    remove_emoji_flag=True,
    normalize_unicode_flag=True,
    remove_punctuation_and_number_flag=True,
    tokenize_vietnamese_flag=True,
    remove_stopwords_flag=True):
    
    if stopwords_set is None:
        with open('stopwords.txt', encoding='utf-8') as f:
            stopwords = set(line.strip() for line in f if line.strip())
    else:
        stopwords = stopwords_set

    if lowercase_flag:
        text = text.lower()
    if remove_emoji_flag:
        text = remove_emoji(text)
    if normalize_unicode_flag:
        text = normalize_unicode(text)
    if remove_punctuation_and_number_flag:
        text = remove_punctuation_and_number(text)
    if tokenize_vietnamese_flag:
        text = normalize_whitespace(text)
        text = tokenize_vietnamese(text)
    if remove_stopwords_flag:
        text = remove_stopwords(text, stopwords)
        text = normalize_whitespace(text)
    return text

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # Để hiển thị tiến độ trong quá trình huấn luyện
import joblib  # Để lưu và tải mô hình

# Đọc dữ liệu
train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/val.csv")
test_df = pd.read_csv("data/test.csv")

# Áp dụng tiền xử lý cho dữ liệu và theo dõi tiến độ
tqdm.pandas(desc="Applying preprocess_text to Train data")
train_df['text'] = train_df['text'].progress_apply(lambda x: preprocess_text(x))  # Áp dụng tiền xử lý cho dữ liệu train

tqdm.pandas(desc="Applying preprocess_text to Validation data")
val_df['text'] = val_df['text'].progress_apply(lambda x: preprocess_text(x))      # Áp dụng tiền xử lý cho dữ liệu validation

tqdm.pandas(desc="Applying preprocess_text to Test data")
test_df['text'] = test_df['text'].progress_apply(lambda x: preprocess_text(x))    # Áp dụng tiền xử lý cho dữ liệu test

# Chia dữ liệu train và validation
X_train = train_df['text']
y_train = train_df['new_label']
X_val = val_df['text']
y_val = val_df['new_label']
X_test = test_df['text']
y_test = test_df['new_label']

# Biến đổi văn bản thành vector sử dụng TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))  # Tạo vector từ 1-gram và 2-gram

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

# Khởi tạo mô hình SVM
svm_model = SVC(kernel='linear', class_weight='balanced')  # class_weight='balanced' giúp cân bằng giữa các lớp

# Huấn luyện mô hình SVM và hiển thị tiến độ
print("Training model...\n")
for epoch in tqdm(range(5), desc="Training Epochs", unit="epoch"):
    svm_model.fit(X_train_tfidf, y_train)

    # Dự đoán trên tập validation
    val_preds = svm_model.predict(X_val_tfidf)
    val_f1 = f1_score(y_val, val_preds, average='macro')
    print(f"Epoch {epoch + 1} - Validation F1 Score: {val_f1:.4f}")

# Lưu mô hình sau khi huấn luyện
model_filename = "models/svm_model.pkl"
joblib.dump(svm_model, model_filename)
print(f"Model saved to {model_filename}")

# Dự đoán trên tập test
test_preds = svm_model.predict(X_test_tfidf)

# Đánh giá kết quả
print("\nTest Classification Report:")
print(classification_report(y_test, test_preds, digits=4))

# Nếu muốn tải lại mô hình đã lưu, bạn có thể làm như sau:
# svm_model_loaded = joblib.load(model_filename)
# print("Model loaded successfully")
# test_preds_loaded = svm_model_loaded.predict(X_test_tfidf)
# print(classification_report(y_test, test_preds_loaded, digits=4))
