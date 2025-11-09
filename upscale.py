# python upscale.py --input p4791e-ext-d000030_3Drecon-ADMBA-P56_pub/3Drecon-ADMBA-P56_annotation.mhd --output atlas_upscaled_v4.zarr --scale 2

import argparse
import dask.array as da
from dask.diagnostics import ProgressBar
import SimpleITK as sitk


def load_mhd_to_dask(mhd_path, chunks=(20, 200, 200)):
    img = sitk.ReadImage(mhd_path)
    arr = sitk.GetArrayFromImage(img)   # -> (Z,Y,X)
    darr = da.from_array(arr, chunks=chunks)
    return darr.astype('uint32')

def upscale_nearest(darr, scale):
    return da.repeat(
        da.repeat(
            da.repeat(darr, scale, axis=0),
            scale, axis=1),
        scale, axis=2)

def save_to_zarr(darr, out_path):
    with ProgressBar():
        darr.to_zarr(out_path, overwrite=True)

def main():
    parser = argparse.ArgumentParser(description="Upscale atlas annotation using nearest-neighbor with Dask.")
    parser.add_argument("--input", required=True, help="Path to annotation .mhd file")
    parser.add_argument("--output", required=True, help="Output folder (Zarr)")
    parser.add_argument("--scale", type=int, default=2, help="Upscaling factor (default: 2)")
    args = parser.parse_args()

    print("🔹 Loading source atlas:", args.input)
    arr = load_mhd_to_dask(args.input)

    print("🔹 Original shape:", arr.shape)
    up = upscale_nearest(arr, args.scale)
    print("🔹 Upscaled shape:", up.shape)

    print("💾 Saving to:", args.output)
    save_to_zarr(up, args.output)

    print("✅ Done.")

if __name__ == "__main__":
    main()
