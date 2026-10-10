
from pathlib import Path
import pandas as pd

# Locate the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load the dataset
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "loan_default_dataset.csv"

df = pd.read_csv(DATA_PATH)

# 1. Dataset dimensions
print("=" * 50)
print("DATASET SHAPE")
print("=" * 50)
print(df.shape)

# 2. Column names
print("\nCOLUMN NAMES")
print(df.columns.tolist())

# 3. Data types and non-null counts
print("\nDATASET INFORMATION")
df.info()

# 4. Missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# 5. Duplicate rows
print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

# 6. Duplicate customer IDs
if "customer_id" in df.columns:
    print("\nDUPLICATE CUSTOMER IDs")
    print(df["customer_id"].duplicated().sum())

# 7. Target distribution
if "default_status" in df.columns:
    print("\nTARGET DISTRIBUTION")
    print(df["default_status"].value_counts(dropna=False))

# 8. Numeric summary
print("\nNUMERICAL SUMMARY")
print(df.describe())

print("\nInitial validation inspection completed.")