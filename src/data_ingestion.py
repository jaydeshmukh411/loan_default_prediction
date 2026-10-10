
from pathlib import Path
import pandas as pd

# Get project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define CSV file path
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "loan_default_dataset.csv"

# Load dataset from CSV
df = pd.read_csv(DATA_PATH)

# Display dataset information
print("Dataset loaded successfully!")
print("Shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())