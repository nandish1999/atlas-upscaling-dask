import SimpleITK as sitk
import pandas as pd
import os

# --- Load the annotation volume ---
volume_path = os.path.join(os.path.dirname(__file__), 
    "p4791e-ext-d000030_3Drecon-ADMBA-P56_pub", 
    "3Drecon-ADMBA-P56_annotation.mhd"
)
img = sitk.ReadImage(volume_path)
arr = sitk.GetArrayFromImage(img)

# --- Load the region lookup CSV ---
csv_path = os.path.join(os.path.dirname(__file__), 
    "p4791e-ext-d000030_3Drecon-ADMBA-P56_pub", 
    "region_ids_ADMBA.csv"
)
df = pd.read_csv(csv_path)

# --- Function to get region name from voxel coordinate ---
def lookup_region(z, y, x):
    region_id = arr[z, y, x]
    region_row = df[df["Region"] == region_id]

    if len(region_row) == 0:
        return f"Unknown region ID: {region_id}"
    
    region_name = region_row["RegionName"].values[0]
    return f"Voxel ({z}, {y}, {x}) → ID {region_id} → {region_name}"

# Try a test lookup:
print(lookup_region(100, 200, 200))
