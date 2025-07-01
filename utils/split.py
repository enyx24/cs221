import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/merged.csv")

print(df["new_label"].value_counts())

df.dropna(inplace=True)

train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['new_label'])
val_df, test_df = train_test_split(temp_df, test_size=2/3, random_state=42, stratify=temp_df['new_label'])

print(train_df["new_label"].value_counts())
print(test_df["new_label"].value_counts())
print(val_df["new_label"].value_counts())

train_df.to_csv("data/full/train.csv", index=False)
val_df.to_csv("data/full/val.csv", index=False)
test_df.to_csv("data/full/test.csv", index=False)

