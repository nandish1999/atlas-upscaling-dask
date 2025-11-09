import SimpleITK as sitk    #to read .mhd volume (annotation)
import dask.array as da     #to handle huge data without loading all into RAM
import zarr     #to save very large 3D data efficiently
import os

# Path to the annotation file
path = os.path.join("p4791e-ext-d000030_3Drecon-ADMBA-P56_pub",
                    "3Drecon-ADMBA-P56_annotation.mhd")

# 1) Load the annotation into memory using SimpleITK 
img = sitk.ReadImage(path)
arr = sitk.GetArrayFromImage(img)   # NumPy array (z, y, x)

print("Original shape:", arr.shape)

# 2) Convert to Dask — Only 20 slices are loaded at a time into memory
d = da.from_array(arr, chunks=(20, arr.shape[1], arr.shape[2]))

# 3) Nearest neighbor upscale ×2 in each axis
d_up = da.repeat(da.repeat(da.repeat(d, 2, axis=0), 2, axis=1), 2, axis=2)

print("Upscaled shape:", d_up.shape)

# 4) Save to Zarr (efficient on-disk storage) .Creates a new chunked (piece-by-piece) file on disk
store = zarr.open("atlas_upscaled.zarr", mode="w",
                  shape=d_up.shape, dtype=d_up.dtype,
                  chunks=(40, 640, 1056))


da.to_zarr(d_up, store, overwrite=True)

print("✅ Upscaling complete.")
print("Output saved to atlas_upscaled.zarr")
