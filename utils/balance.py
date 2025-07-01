import pandas as pd
import random
from sklearn.utils import shuffle
import nlpaug.augmenter.word as naw
import nltk
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('omw-1.4')

aug = naw.SynonymAug(aug_src='wordnet', lang='vie')
df = pd.read_csv("data/merged.csv")

df_pos = df[df["new_label"] == 1].copy()
df_neg = df[df["new_label"] == 0].copy()

pos_vihsd = df_pos[df_pos["origin"] == "vihsd"]
pos_victsd = df_pos[df_pos["origin"] == "victsd"]

def augment(df, origin_name, n_aug):
    aug = naw.RandomWordAug(action="swap")
    samples = df.sample(n=min(len(df), n_aug), replace=True, random_state=42).copy()
    texts = samples['text'].astype(str).tolist()

    augmented_texts = []
    for text in texts:
        try:
            aug_text = aug.augment(text)
        except Exception as e:
            print(f"⚠️ Augment error: {e} -> dùng text gốc.")
            aug_text = text  # Nếu lỗi thì fallback về text gốc
        augmented_texts.append(aug_text)

    samples['text'] = augmented_texts
    samples['origin'] = f"gen-{origin_name}"
    return samples


aug_pos_vihsd = augment(pos_vihsd, "vihsd", int(0.8 * 2000))
aug_pos_victsd = augment(pos_victsd, "victsd", int(0.2 * 2000))
df_pos_augmented = pd.concat([df_pos, aug_pos_vihsd, aug_pos_victsd])

# Drop class 0 xuống còn ~18000 (từ 36k), giữ lại khoảng:
# - 75% từ vihsd
# - 25% từ victsd
neg_vihsd = df_neg[df_neg["origin"] == "vihsd"].sample(n=int(18000*0.75), random_state=42)
neg_victsd = df_neg[df_neg["origin"] == "victsd"].sample(n=int(18000*0.25), random_state=42)
df_neg_dropped = pd.concat([neg_vihsd, neg_victsd])

# Gộp lại
df_final = pd.concat([df_pos_augmented, df_neg_dropped])
df_final = shuffle(df_final, random_state=42).reset_index(drop=True)

# Xuất ra file
df_final.to_csv("data/balanced.csv", index=False)

# Thống kê kiểm tra
print("Stats:")
print(df_final["new_label"].value_counts())
print(df_final["origin"].value_counts())
