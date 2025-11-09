import SimpleITK as sitk

path = "p4791e-ext-d000030_3Drecon-ADMBA-P56_pub/3Drecon-ADMBA-P56_annotation.mhd"
img = sitk.ReadImage(path)
arr = sitk.GetArrayFromImage(img)

print("Shape (z,y,x):", arr.shape)
print("Data type:", arr.dtype)
print("Small sample:", arr[100, 200, 200])
