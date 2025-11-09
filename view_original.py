import napari
import SimpleITK as sitk
import numpy as np

# Path to original annotation atlas
path = "p4791e-ext-d000030_3Drecon-ADMBA-P56_pub/3Drecon-ADMBA-P56_annotation.mhd"

print("Loading original atlas...")

# Load the image using SimpleITK
img = sitk.ReadImage(path)

# Convert to numpy (z, y, x)
arr = sitk.GetArrayFromImage(img)

print("Shape:", arr.shape)
print("Data type:", arr.dtype)

# Launch napari
viewer = napari.Viewer()
viewer.add_labels(arr, name="original_annotation")
napari.run()
