# If voxel (100,200,200) was Putamen before upscaling,
# Then voxel (200,400,400) should also be Putamen after upscaling.
# Checks if the brain region label remain identical after upscaling?

import SimpleITK as sitk
import dask.array as da
import zarr
import numpy as np

# Load original annotation
img = sitk.ReadImage("p4791e-ext-d000030_3Drecon-ADMBA-P56_pub/3Drecon-ADMBA-P56_annotation.mhd")
orig = sitk.GetArrayFromImage(img)

# Load upscaled atlas
upscaled = da.from_zarr("atlas_upscaled.zarr")

# Pick a coordinate to check
z, y, x = 100, 200, 200

orig_value = orig[z, y, x]
up_value = upscaled[z*2, y*2, x*2].compute()

print(f"Original @ ({z},{y},{x}) = {orig_value}")
print(f"Upscaled @ ({z*2},{y*2},{x*2}) = {up_value}")
