import pandas as pd

df_dev = pd.read_csv("vihsd/dev.csv")
df_test = pd.read_csv("vihsd/test.csv")
df_train = pd.read_csv("vihsd/train.csv")

df_dev = df_dev.rename(columns={'label_id': 'label'})
df_test = df_test.rename(columns={'label_id': 'label'})
df_train = df_train.rename(columns={'label_id': 'label'})
df_dev = df_dev.rename(columns={'free_text': 'text'})
df_test = df_test.rename(columns={'free_text': 'text'})
df_train = df_train.rename(columns={'free_text': 'text'})


vihsd = pd.concat([df_dev, df_test, df_train], ignore_index=True)
vihsd['origin'] = 'vihsd'


df_val = pd.read_csv("ViCTSD/ViCTSD_valid.csv")
df_test = pd.read_csv("ViCTSD/ViCTSD_test.csv")
df_train = pd.read_csv("ViCTSD/ViCTSD_train.csv")

df_dev = df_dev.rename(columns={'Toxicity': 'label'})
df_test = df_test.rename(columns={'Toxicity': 'label'})
df_train = df_train.rename(columns={'Toxicity': 'label'})
df_dev = df_dev.rename(columns={'Comment': 'text'})
df_test = df_test.rename(columns={'Comment': 'text'})
df_train = df_train.rename(columns={'Comment': 'text'})


victsd = pd.concat([df_dev, df_test, df_train], ignore_index=True)
victsd['origin'] = 'victsd'
victsd.drop(columns=['Unnamed: 0'], inplace=True)
victsd.drop(columns=['Constructiveness'], inplace=True)
victsd.drop(columns=['Title'], inplace=True)
victsd.drop(columns=['Topic'], inplace=True)

df = pd.concat([vihsd, victsd], ignore_index=True)
df['new_label'] = df['label'].apply(lambda x: 1 if x == 2 else x)
df.to_csv("data/merged.csv", index=False)

print(df['label'].value_counts())
print(df['new_label'].value_counts())

df_sample = df.sample(n=75, random_state=42)
df_sample.reset_index(inplace=True)
df_sample.to_csv("data/random.csv", index=False)

