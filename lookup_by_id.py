import pandas as pd
import os


csv_path = os.path.join(os.path.dirname(__file__), 
    "p4791e-ext-d000030_3Drecon-ADMBA-P56_pub", 
    "region_ids_ADMBA.csv"
)

# Load region table
df = pd.read_csv(csv_path)

print("\n=== Region ID Lookup Tool ===")
print("Enter a region ID (e.g., 15857) or type 'exit' to quit.\n")

while True:
    user_input = input("Enter Region ID: ")

    if user_input.lower() in ["exit", "quit", "q"]:
        print("\nGoodbye!\n")
        break

    # Check if numeric
    if not user_input.isdigit():
        print("⚠️  Please enter a numeric region ID.\n")
        continue

    region_id = int(user_input)

    row = df[df["Region"] == region_id]

    if len(row) == 0:
        print(f"❌ No region found for ID {region_id}\n")
    else:
        name = row["RegionName"].values[0]
        abbr = row["RegionAbbr"].values[0]
        level = row["Level"].values[0]
        print(f"✅ Region ID {region_id} = {name} ({abbr}), Level {level}\n")
