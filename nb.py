import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from utils.preprocess import preprocess_text

def train(train_df):
    # Tiền xử lý văn bản
    train_df = train_df.dropna(subset=["text", "new_label"])
    texts = train_df["text"].apply(preprocess_text)
    labels = train_df["new_label"]

    # Vector hóa
    vectorizer = CountVectorizer(max_features=10000)
    X_train = vectorizer.fit_transform(texts)

    # Huấn luyện mô hình
    model = MultinomialNB()
    model.fit(X_train, labels)

    return model, vectorizer

def test(model, vectorizer, test_df, print_report=True):
    # Tiền xử lý
    test_df = test_df.dropna(subset=["text", "new_label"])
    texts = test_df["text"].apply(preprocess_text)
    labels = test_df["new_label"]

    X_test = vectorizer.transform(texts)
    preds = model.predict(X_test)

    if print_report:
        print(classification_report(labels, preds, digits=4))
    return preds, labels


def __main__():
    # Đọc dữ liệu
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    # Huấn luyện mô hình
    model, vectorizer = train(train_df)

    # Dự đoán trên tập kiểm tra
    preds, labels = test(model, vectorizer, test_df)

if __name__ == "__main__":
    __main__()