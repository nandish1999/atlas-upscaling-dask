import napari
import zarr
import pandas as pd
import numpy as np
import os

ATLAS_ZARR = "atlas_upscaled_v4.zarr"
CSV = os.path.join(os.path.dirname(__file__), 
    "p4791e-ext-d000030_3Drecon-ADMBA-P56_pub", 
    "region_ids_ADMBA.csv"
)

# load volumes
arr = zarr.open(ATLAS_ZARR, mode='r')
df = pd.read_csv(CSV).set_index("Region")

# create viewer
viewer = napari.Viewer()
layer = viewer.add_labels(arr, name="Atlas")

@layer.mouse_drag_callbacks.append
def on_click(layer, event):
    pos = tuple(int(x) for x in layer.world_to_data(event.position))
    region_id = layer.data[pos]
    if region_id in df.index:
        print(f"Clicked: ID {region_id} → {df.loc[region_id]['RegionName']}")
    else:
        print(f"Unknown ID {region_id}")

napari.run()

