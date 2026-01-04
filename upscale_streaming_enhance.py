#!/usr/bin/env python3
"""
upscale_streaming_enhance.py

Enhancements included:
1) Multi-scale OME-Zarr pyramid output
2) Outline-only mode for labels (edge voxels only)
4) CLI validation + safety guards:
   - --dry-run (prints plan, no compute / no output written)
   - output size estimation with --max-gb limit
   - scale guard (blocks --scale > 20 unless --force)
   - pyramid confirmation (blocks --pyramid-levels > 1 unless --force)

Notes:
- This script is meant to be *safe* on shared / institutional clusters.
- For label maps, pyramid downsampling here is simple decimation (every 2nd voxel).
"""

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

    if "DimSize" not in meta or "ElementType" not in meta or "ElementDataFile" not in meta:
        raise ValueError("Invalid MHD file: missing required fields (DimSize, ElementType, ElementDataFile).")

    meta["DimSize"] = [int(x) for x in meta["DimSize"]]
    return meta

def mhd_memmap(mhd_path: str):
    meta = parse_mhd(mhd_path)
    X, Y, Z = meta["DimSize"]  # MHD usually stores as X Y Z
    if meta["ElementType"] not in _MHD_DTYPE:
        raise ValueError(f"Unsupported ElementType: {meta['ElementType']}")
    dtype = _MHD_DTYPE[meta["ElementType"]]
    raw_path = os.path.join(os.path.dirname(mhd_path), meta["ElementDataFile"])

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"RAW file not found: {raw_path}")

    if str(meta.get("ByteOrderMSB", "False")).lower() == "true":
        dtype = dtype.newbyteorder(">")

    # We read as (Z, Y, X)
    mm = np.memmap(raw_path, mode="r", dtype=dtype, shape=(Z, Y, X))
    return mm, meta

def choose_chunks(shape_zyx, bpp: int, target_mb: int):
    """
    Choose (cz, cy, cx) aiming for ~target_mb per chunk, bounded by shape and 512 in y/x.
    """
    target = int(target_mb) * 1024 * 1024
    z, y, x = shape_zyx
    cz, cy, cx = 16, min(512, y), min(512, x)
    denom = max(1, cz * cy * cx * bpp)
    scale = (target / denom) ** (1/3)

    cz2 = max(1, min(z, int(max(1, round(cz * scale)))))
    cy2 = max(1, min(y, int(max(1, round(cy * scale)))))
    cx2 = max(1, min(x, int(max(1, round(cx * scale)))))
    return (cz2, cy2, cx2)

def _make_compressor(name: str):
    name = (name or "zstd").lower()
    if name == "zstd":
        return Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    if name == "lz4":
        return Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)
    if name in ("none", "null", "no"):
        return None
    raise ValueError(f"Unknown compressor: {name}")


# ----------------------------
# Enhancement #2: Outline mode
# ----------------------------
def apply_outline(d: da.Array) -> da.Array:
    """
    Keep only label boundaries (6-neighborhood).
    Voxels whose value differs from any neighbor are kept.
    """
    return d * (
        (d != da.roll(d,  1, 0)) |
        (d != da.roll(d, -1, 0)) |
        (d != da.roll(d,  1, 1)) |
        (d != da.roll(d, -1, 1)) |
        (d != da.roll(d,  1, 2)) |
        (d != da.roll(d, -1, 2))
    )


# ----------------------------
# Enhancement #3: Pyramid builder
# ----------------------------
def build_pyramid(base: da.Array, levels: int):
    """
    For label maps, we do simple decimation by stride-2.
    (This keeps labels stable-ish and is cheap; not "averaging".)
    """
    levels = int(levels)
    if levels < 1:
        raise ValueError("--pyramid-levels must be >= 1")

    pyr = [base]
    for _ in range(1, levels):
        prev = pyr[-1]
        down = prev[::2, ::2, ::2]
        pyr.append(down)
    return pyr


# ----------------------------
# Enhancement #4: Safety guards
# ----------------------------
def estimate_output_gb(
    shape_zyx,
    dtype,
    scale: int,
    pyramid_levels: int,
    include_pyramid_overhead: bool = True,
) -> float:
    """
    Estimate total stored bytes for the *upscaled* base level and optionally the pyramid overhead.

    Base level bytes = Z*Y*X*(scale^3)*dtype_size
    Pyramid overhead ~ sum_{i=1..L-1} 1/(2^3)^i = 1/7 (for infinite), so ~ (1 + 1/7) for many levels.
    For small L, this is a slight overestimate, which is fine for safety.
    """
    z, y, x = [int(v) for v in shape_zyx]
    scale = int(scale)
    itemsize = np.dtype(dtype).itemsize
    base_bytes = z * y * x * (scale ** 3) * itemsize

    if include_pyramid_overhead and int(pyramid_levels) > 1:
        base_bytes = base_bytes * (1.0 + 1.0 / 7.0)

    return base_bytes / (1024.0 ** 3)


def print_plan(shape_zyx, dtype, scale, chunk_mb, chunks, mode, pyramid_levels, out_path, compressor, est_gb):
    out_shape = (int(shape_zyx[0] * scale), int(shape_zyx[1] * scale), int(shape_zyx[2] * scale))
    print("\n🔍 Execution plan")
    print(f"  Input shape (z,y,x) : {tuple(map(int, shape_zyx))}")
    print(f"  Input dtype         : {np.dtype(dtype)} ({np.dtype(dtype).itemsize} bytes/voxel)")
    print(f"  Scale factor        : {int(scale)}")
    print(f"  Output shape (z,y,x): {out_shape}")
    print(f"  Mode                : {mode}")
    print(f"  Pyramid levels      : {int(pyramid_levels)}")
    print(f"  Chunk target (MB)   : {int(chunk_mb)}")
    print(f"  Chunks (z,y,x)      : {tuple(map(int, chunks))}")
    print(f"  Compressor          : {compressor}")
    print(f"  Output path         : {out_path}")
    print(f"  Estimated output    : {est_gb:.2f} GB")
    print("")


# ----------------------------
# Writer (OME-Zarr pyramid)
# ----------------------------
def write_ome_zarr_pyramid(pyramid, out_dir, meta, scale, compressor):
    """
    Writes multiscale OME-Zarr pyramid levels at paths "0", "1", ... with OME multiscales metadata.
    """
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    store = zarr.DirectoryStore(out_dir)
    root = zarr.group(store=store)

    datasets = []
    spacing = meta.get("ElementSpacing", [1, 1, 1])  # typically [sx, sy, sz] (x,y,z)

    # Validate spacing
    if not (isinstance(spacing, (list, tuple)) and len(spacing) == 3):
        spacing = [1, 1, 1]

    for i, arr in enumerate(pyramid):
        print(f"Writing pyramid level {i}, shape={arr.shape}")

        # Ensure chunk-size config won't try to assemble tiny chunks in memory
        try:
            csz = arr.chunksize
            if csz is not None:
                out_chunk_bytes = int(np.prod(csz)) * np.dtype(arr.dtype).itemsize
                dask.config.set({"array.chunk-size": max(out_chunk_bytes, 32 * 1024 * 1024)})
        except Exception:
            pass

        z = root.create_dataset(
            str(i),
            shape=arr.shape,
            chunks=arr.chunksize,
            dtype=arr.dtype,
            compressor=compressor,
        )
        with ProgressBar():
            da.store(arr, z)

        # OME-Zarr coordinateTransformations are for (z,y,x) here
        # original spacing is (x,y,z); after upscaling, voxel spacing decreases by /scale
        # pyramid level i increases voxel spacing by *2^i
        level_scale_factor = (2 ** i)
        datasets.append({
            "path": str(i),
            "coordinateTransformations": [{
                "type": "scale",
                "scale": [
                    (spacing[2] * level_scale_factor) / scale,  # z
                    (spacing[1] * level_scale_factor) / scale,  # y
                    (spacing[0] * level_scale_factor) / scale,  # x
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

    # Helpful: tiny metadata file for debugging / provenance
    info = {
        "source_mhd": os.path.abspath(getattr(meta, "source", "")) if isinstance(meta, dict) else "",
        "element_spacing_xyz": meta.get("ElementSpacing", None) if isinstance(meta, dict) else None,
        "scale": int(scale),
        "pyramid_levels": int(len(pyramid)),
    }
    try:
        with open(os.path.join(out_dir, ".atlas_upscale_meta.json"), "w") as f:
            json.dump(info, f, indent=2)
    except Exception:
        pass


# ----------------------------
# Main pipeline
# ----------------------------
def main():
    p = argparse.ArgumentParser("Streaming multi-scale OME-Zarr label upscaler (with safety guards)")
    p.add_argument("--input", required=True, help="Path to annotation .mhd")
    p.add_argument("--output", required=True, help="Output directory (.ome.zarr folder)")

    p.add_argument("--scale", type=int, default=2, help="Integer upscaling factor")
    p.add_argument("--chunk-mb", type=int, default=128, help="Target chunk size in MB")

    p.add_argument("--compressor", default="zstd", choices=["zstd", "lz4", "none"],
                   help="Zarr compressor")
    p.add_argument("--mode", choices=["fill", "outline"], default="fill",
                   help="fill = full regions, outline = edges only")

    p.add_argument("--pyramid-levels", type=int, default=1,
                   help="Number of multiscale pyramid levels (>=1). Level 0 is full-res.")

    # Enhancement #4 flags
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan and exit (no compute, no output written)")
    p.add_argument("--force", action="store_true",
                   help="Bypass safety guards (expert use)")
    p.add_argument("--max-gb", type=float, default=500.0,
                   help="Maximum allowed estimated output size in GB (blocked unless --force)")

    args = p.parse_args()

    # --- Load MHD header + memmap (cheap; no full RAM materialization) ---
    mm, meta = mhd_memmap(args.input)
    shape_zyx = mm.shape
    dtype = mm.dtype

    # --- Plan / estimate ---
    bpp = dtype.itemsize
    chunks = choose_chunks(shape_zyx, bpp, args.chunk_mb)

    est_gb = estimate_output_gb(
        shape_zyx=shape_zyx,
        dtype=dtype,
        scale=args.scale,
        pyramid_levels=args.pyramid_levels,
        include_pyramid_overhead=True,
    )

    print_plan(
        shape_zyx=shape_zyx,
        dtype=dtype,
        scale=args.scale,
        chunk_mb=args.chunk_mb,
        chunks=chunks,
        mode=args.mode,
        pyramid_levels=args.pyramid_levels,
        out_path=args.output,
        compressor=args.compressor,
        est_gb=est_gb,
    )

    # --- Guards ---
    if args.scale > 20 and not args.force:
        raise RuntimeError("❌ --scale > 20 is blocked. Use --force to override.")

    if est_gb > args.max_gb and not args.force:
        raise RuntimeError(
            f"❌ Estimated output {est_gb:.1f} GB exceeds --max-gb {args.max_gb:.1f} GB. "
            f"Use --force to override, or lower --scale / --pyramid-levels."
        )

    if args.pyramid_levels > 1 and not args.force:
        raise RuntimeError("❌ --pyramid-levels > 1 requires --force (explicit confirmation).")

    if args.dry_run:
        print("🟡 --dry-run enabled. Exiting without computation.")
        return

    # --- Build base dask array (still lazy) ---
    base = da.from_array(mm, chunks=chunks, asarray=False)

    if args.scale != 1:
        base = da.repeat(base, args.scale, axis=0)
        base = da.repeat(base, args.scale, axis=1)
        base = da.repeat(base, args.scale, axis=2)

    if args.mode == "outline":
        print("Applying outline (edge-only) mode")
        base = apply_outline(base)

    # --- Build pyramid (lazy) ---
    pyramid = build_pyramid(base, args.pyramid_levels)

    # --- Write ---
    comp = _make_compressor(args.compressor)
    write_ome_zarr_pyramid(pyramid, args.output, meta, args.scale, comp)

    print("✅ Multi-scale OME-Zarr written:", args.output)


if __name__ == "__main__":
    main()