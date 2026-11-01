"""Clean the data and write stratified train/test splits to the working directory."""
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
TARGET = "ProdTaken"

df = pd.read_csv(DATA_PATH)
print("Loaded:", df.shape)

# 1. Drop identifier / index columns that carry no predictive signal
df = df.drop(columns=[c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns])

# 2. Fix inconsistent labels
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

# 3. Remove exact duplicate customers (same features, different ID) so copies cannot
#    land in both train and test and inflate the scores
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Dropped {before - len(df)} duplicate rows")

# 4. Strip whitespace from text columns and impute any gaps
cat_cols = df.select_dtypes(exclude="number").columns
for c in cat_cols:
    df[c] = df[c].astype(str).str.strip()
num_cols = df.drop(columns=[TARGET]).select_dtypes(include="number").columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# 5. Stratified split (target is ~19% positive)
X, y = df.drop(columns=[TARGET]), df[TARGET]
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print(f"Train: {Xtrain.shape}  positives={ytrain.mean():.3f}")
print(f"Test : {Xtest.shape}  positives={ytest.mean():.3f}")
