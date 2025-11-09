import pandas as pd
import os

# Find folder where this script is located
base_path = os.path.dirname(__file__)

# Build correct path to CSV
csv_path = os.path.join(base_path, "p4791e-ext-d000030_3Drecon-ADMBA-P56_pub", "region_ids_ADMBA.csv")

# Load CSV
df = pd.read_csv(csv_path)

print(df.head())  #show me the first 5 rows of the dataset
print("\nNumber of regions:", len(df))
