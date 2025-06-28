import pandas as pd

# Read all 3 files into dataframes
df_dev = pd.read_csv("dev.csv")
df_test = pd.read_csv("test.csv")
df_train = pd.read_csv("train.csv")

# Print number of rows for each dataframe
print("dev.csv rows:", len(df_dev))
print("test.csv rows:", len(df_test))
print("train.csv rows:", len(df_train))

# Print column labels for each dataframe
print("dev.csv columns:", df_dev.columns.tolist())
print("test.csv columns:", df_test.columns.tolist())
print("train.csv columns:", df_train.columns.tolist())

# Print label counts for each dataframe (assuming 'label' column exists)
if 'label_id' in df_dev.columns:
    print("dev.csv label counts:\n", df_dev['label_id'].value_counts())
if 'label_id' in df_test.columns:
    print("test.csv label counts:\n", df_test['label_id'].value_counts())
if 'label_id' in df_train.columns:
    print("train.csv label counts:\n", df_train['label_id'].value_counts())

