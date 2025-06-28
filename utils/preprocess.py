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
