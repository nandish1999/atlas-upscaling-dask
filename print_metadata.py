import SimpleITK as sitk
import sys

mhd = sys.argv[1]
img = sitk.ReadImage(mhd)
spacing = img.GetSpacing()
size = img.GetSize()

print("=== Metadata Report ===")
print(f"File: {mhd}")
print(f"Voxel size (µm): {spacing}")
print(f"Dimensions (Z,Y,X): {size}")
print(f"Volume size (mm): {[s*z/1000 for s,z in zip(size, spacing)]}")
