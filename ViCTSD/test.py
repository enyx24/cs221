import pandas as pd

df_train = pd.read_csv("ViCTSD_train.csv")
df_test = pd.read_csv("ViCTSD_test.csv")
df_val = pd.read_csv("ViCTSD_valid.csv")

print("train rows:", len(df_train))
print("test rows:", len(df_test))
print("dev rows:", len(df_val))

if 'Toxicity' in df_train.columns:
    print("train label counts:\n", df_train['Toxicity'].value_counts())
if 'Toxicity' in df_test.columns:
    print("test label counts:\n", df_test['Toxicity'].value_counts())
if 'Toxicity' in df_val.columns:
    print("val label counts:\n", df_val['Toxicity'].value_counts())
