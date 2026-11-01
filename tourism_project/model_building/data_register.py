"""Register the dataset: confirm the CSV inside the repo has the expected schema."""
import sys
import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"

EXPECTED = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier", "DurationOfPitch",
    "Occupation", "Gender", "NumberOfPersonVisiting", "NumberOfFollowups",
    "ProductPitched", "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting",
    "Designation", "MonthlyIncome",
]

df = pd.read_csv(DATA_PATH)
missing = [c for c in EXPECTED if c not in df.columns]
if missing:
    sys.exit(f"Dataset is missing expected columns: {missing}")

print(f"Dataset registered from {DATA_PATH}")
print(f"Shape            : {df.shape}")
print(f"Target balance   : {df['ProdTaken'].value_counts(normalize=True).round(3).to_dict()}")
print(f"Missing values   : {int(df.isna().sum().sum())}")
print(f"Duplicate IDs    : {int(df['CustomerID'].duplicated().sum())}")
