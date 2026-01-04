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

def parse_mhd(mhd_path: str) -> dict:
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

    if "DimSize" not in meta or "ElementDataFile" not in meta or "ElementType" not in meta:
        raise ValueError("MHD missing required fields.")

    meta["DimSize"] = [int(x) for x in meta["DimSize"]]
    return meta


def mhd_memmap(mhd_path: str):
    meta = parse_mhd(mhd_path)
    X, Y, Z = meta["DimSize"]
    dtype = _MHD_DTYPE[meta["ElementType"]]
    raw_path = os.path.join(os.path.dirname(mhd_path), meta["ElementDataFile"])

    if str(meta.get("ByteOrderMSB", "False")).lower() == "true":
        dtype = dtype.newbyteorder(">")

    mm = np.memmap(raw_path, mode="r", dtype=dtype, shape=(Z, Y, X), order="C")
    return mm, meta


def choose_chunks(shape_zyx, bytes_per_voxel, target_chunk_mb=128):
    z, y, x = shape_zyx
    target_bytes = int(target_chunk_mb * 1024 * 1024)

    cz, cy, cx = 16, min(512, y), min(512, x)
    chunk_bytes = cz * cy * cx * bytes_per_voxel

    if chunk_bytes > 0:
        scale = (target_bytes / chunk_bytes) ** (1/3)
        cz = max(1, min(z, int(round(cz * scale))))
        cy = max(1, min(y, int(round(cy * scale))))
        cx = max(1, min(x, int(round(cx * scale))))

    return (cz, cy, cx)


def _make_compressor(name: str):
    if name == "zstd":
        return Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    if name == "lz4":
        return Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)
    if name == "none":
        return None
    raise ValueError(f"Unknown compressor: {name}")


# ----------------------------
# Enhancement #2: outline mode
# ----------------------------
def apply_outline_mode(darr: da.Array) -> da.Array:
    dzp = da.roll(darr,  1, axis=0)
    dzm = da.roll(darr, -1, axis=0)
    dyp = da.roll(darr,  1, axis=1)
    dym = da.roll(darr, -1, axis=1)
    dxp = da.roll(darr,  1, axis=2)
    dxm = da.roll(darr, -1, axis=2)

    edge = (
        (darr != dzp) |
        (darr != dzm) |
        (darr != dyp) |
        (darr != dym) |
        (darr != dxp) |
        (darr != dxm)
    )

    return darr * edge


# ----------------------------
# Core: build upscaled array
# ----------------------------
def build_upscaled_dask_from_mhd(mhd_path, scale, chunk_mb, mode):
    mm, meta = mhd_memmap(mhd_path)
    zyx = mm.shape
    dtype = mm.dtype.newbyteorder("=")
    bpp = mm.dtype.itemsize

    print(f"Source shape (z,y,x): {zyx}, dtype={dtype}, spacing={meta.get('ElementSpacing')}")

    chunks = choose_chunks(zyx, bpp, chunk_mb)
    print(f"Using input chunks (z,y,x): {chunks}")

    darr = da.from_array(mm, chunks=chunks, asarray=False)

    if scale != 1:
        darr = da.repeat(darr, scale, axis=0)
        darr = da.repeat(darr, scale, axis=1)
        darr = da.repeat(darr, scale, axis=2)

    if mode == "outline":
        print("Applying outline (edge-only) mode")
        darr = apply_outline_mode(darr)

    out_shape = tuple(s * scale for s in zyx)
    print(f"Upscaled shape (z,y,x): {out_shape}")

    return darr, meta, chunks, zyx, out_shape, dtype


# ----------------------------
# Writers
# ----------------------------
def write_zarr_or_omezarr(up, out_dir, chunks, compressor, meta, src,
                          scale, in_shape, out_shape, dtype, fmt):

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    out_chunk_bytes = np.prod(chunks) * np.dtype(dtype).itemsize
    dask.config.set({"array.chunk-size": max(out_chunk_bytes, 32 * 1024 * 1024)})

    store = zarr.DirectoryStore(out_dir)
    with ProgressBar():
        up.rechunk(chunks).to_zarr(store, overwrite=True, compressor=compressor)

    info = {
        "source": os.path.abspath(src),
        "scale": scale,
        "mode": "outline" if "outline" in fmt else "fill",
        "source_shape_zyx": list(in_shape),
        "output_shape_zyx": list(out_shape),
        "chunks_zyx": list(chunks),
        "dtype": str(dtype),
        "format": fmt,
    }

    with open(os.path.join(out_dir, ".atlas_upscale_meta.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"✅ Finished. {fmt} written to: {out_dir}")


# ----------------------------
# Main pipeline
# ----------------------------
def upscale_streaming(mhd_path, output, fmt, scale, chunk_mb, compressor, mode):
    up, meta, chunks, in_shape, out_shape, dtype = build_upscaled_dask_from_mhd(
        mhd_path, scale, chunk_mb, mode
    )

    comp = _make_compressor(compressor)

    if fmt in ("zarr", "ome.zarr"):
        write_zarr_or_omezarr(
            up, output, chunks, comp, meta, mhd_path,
            scale, in_shape, out_shape, dtype, fmt
        )
        return

    raise ValueError("Outline mode currently supported for zarr / ome.zarr only.")


def main():
    p = argparse.ArgumentParser("Streaming atlas upscaler with fill / outline modes")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--format", default="zarr", choices=["zarr", "ome.zarr"])
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--chunk-mb", type=int, default=128)
    p.add_argument("--compressor", default="zstd", choices=["zstd", "lz4", "none"])
    p.add_argument("--mode", default="fill", choices=["fill", "outline"],
                   help="fill = full labels, outline = edges only")

    args = p.parse_args()

    upscale_streaming(
        mhd_path=args.input,
        output=args.output,
        fmt=args.format,
        scale=args.scale,
        chunk_mb=args.chunk_mb,
        compressor=args.compressor,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()