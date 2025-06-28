import re
import emoji
import unidecode
from underthesea import word_tokenize
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

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
def preprocess_text(text, stopwords):
    text = text.lower()
    text = remove_emoji(text)
    text = normalize_unicode(text)
    text = remove_punctuation_and_number(text)
    text = normalize_whitespace(text)
    text = tokenize_vietnamese(text)
    text = remove_stopwords(text, stopwords)
    text = normalize_whitespace(text)
    return text
