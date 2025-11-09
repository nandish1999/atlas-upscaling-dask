import napari
import dask.array as da
import zarr

print("Loading upscaled atlas...")

# Path to your upscaled Zarr folder
path = "atlas_upscaled_v4.zarr"   

# Open the Zarr as a Dask array (memory-safe)
arr = da.from_zarr(path)

print("Shape:", arr.shape)
print("Data type:", arr.dtype)

# Launch napari
viewer = napari.Viewer()
viewer.add_labels(arr, name="upscaled_annotation")
napari.run()
