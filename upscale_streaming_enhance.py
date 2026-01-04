import argparse, os, re, json, shutil
import numpy as np
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
import zarr
from numcodecs import Blosc

# ----------------------------
# MHD parsing + memmap helpers
# ----------------------------
_MHD_DTYPE = {
    "MET_UCHAR": np.uint8,
    "MET_CHAR": np.int8,
    "MET_USHORT": np.uint16,
    "MET_SHORT": np.int16,
    "MET_UINT": np.uint32,
    "MET_INT": np.int32,
    "MET_FLOAT": np.float32,
    "MET_DOUBLE": np.float64,
}

def parse_mhd(mhd_path):
    meta = {}
    with open(mhd_path, "r") as f:
        for line in f:
            if "=" not in line:
                continue
            k, v = [s.strip() for s in line.split("=", 1)]
            if k in ("DimSize", "ElementSpacing"):
                meta[k] = [float(x) for x in re.split(r"[ ,]", v) if x]
            else:
                meta[k] = v

    if "DimSize" not in meta or "ElementType" not in meta or "ElementDataFile" not in meta:
        raise ValueError("Invalid MHD file")

    meta["DimSize"] = [int(x) for x in meta["DimSize"]]
    return meta

def mhd_memmap(mhd_path):
    meta = parse_mhd(mhd_path)
    X, Y, Z = meta["DimSize"]
    dtype = _MHD_DTYPE[meta["ElementType"]]
    raw_path = os.path.join(os.path.dirname(mhd_path), meta["ElementDataFile"])

    if str(meta.get("ByteOrderMSB", "False")).lower() == "true":
        dtype = dtype.newbyteorder(">")

    mm = np.memmap(raw_path, mode="r", dtype=dtype, shape=(Z, Y, X))
    return mm, meta

def choose_chunks(shape, bpp, target_mb):
    target = target_mb * 1024 * 1024
    cz, cy, cx = 16, min(512, shape[1]), min(512, shape[2])
    scale = (target / (cz * cy * cx * bpp)) ** (1/3)
    return (
        max(1, min(shape[0], int(cz * scale))),
        max(1, min(shape[1], int(cy * scale))),
        max(1, min(shape[2], int(cx * scale))),
    )

def _make_compressor(name):
    if name == "zstd":
        return Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    if name == "lz4":
        return Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)
    return None

# ----------------------------
# Outline mode
# ----------------------------
def apply_outline(d):
    return d * (
        (d != da.roll(d, 1, 0)) |
        (d != da.roll(d, -1, 0)) |
        (d != da.roll(d, 1, 1)) |
        (d != da.roll(d, -1, 1)) |
        (d != da.roll(d, 1, 2)) |
        (d != da.roll(d, -1, 2))
    )

# ----------------------------
# Pyramid builder
# ----------------------------
def build_pyramid(base, levels):
    pyr = [base]
    for i in range(1, levels):
        prev = pyr[-1]
        down = prev[::2, ::2, ::2]
        pyr.append(down)
    return pyr

# ----------------------------
# Writer
# ----------------------------
def write_ome_zarr_pyramid(pyramid, out_dir, meta, scale, compressor):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    store = zarr.DirectoryStore(out_dir)
    root = zarr.group(store=store)

    datasets = []
    spacing = meta.get("ElementSpacing", [1,1,1])

    for i, arr in enumerate(pyramid):
        print(f"Writing pyramid level {i}, shape={arr.shape}")
        z = root.create_dataset(
            str(i),
            shape=arr.shape,
            chunks=arr.chunksize,
            dtype=arr.dtype,
            compressor=compressor,
        )
        with ProgressBar():
            da.store(arr, z)

        scale_factor = (2 ** i)
        datasets.append({
            "path": str(i),
            "coordinateTransformations": [{
                "type": "scale",
                "scale": [
                    spacing[2] * scale_factor / scale,
                    spacing[1] * scale_factor / scale,
                    spacing[0] * scale_factor / scale,
                ]
            }]
        })

    root.attrs["multiscales"] = [{
        "version": "0.4",
        "name": "labels",
        "axes": [
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ],
        "datasets": datasets
    }]
    root.attrs["image-label"] = True

# ----------------------------
# Main pipeline
# ----------------------------
def main():
    p = argparse.ArgumentParser("Streaming multi-scale OME-Zarr label upscaler")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--chunk-mb", type=int, default=128)
    p.add_argument("--compressor", default="zstd")
    p.add_argument("--mode", choices=["fill", "outline"], default="fill")
    p.add_argument("--pyramid-levels", type=int, default=1)
    args = p.parse_args()

    mm, meta = mhd_memmap(args.input)
    bpp = mm.dtype.itemsize
    chunks = choose_chunks(mm.shape, bpp, args.chunk_mb)

    base = da.from_array(mm, chunks=chunks)
    if args.scale != 1:
        base = da.repeat(base, args.scale, 0)
        base = da.repeat(base, args.scale, 1)
        base = da.repeat(base, args.scale, 2)

    if args.mode == "outline":
        base = apply_outline(base)

    pyramid = build_pyramid(base, args.pyramid_levels)
    comp = _make_compressor(args.compressor)

    write_ome_zarr_pyramid(pyramid, args.output, meta, args.scale, comp)
    print("✅ Multi-scale OME-Zarr written:", args.output)

if __name__ == "__main__":
    main()