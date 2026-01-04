import argparse, os, re, json, math, shutil
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
        raise ValueError("MHD missing required fields (DimSize, ElementType, ElementDataFile).")

    meta["DimSize"] = [int(x) for x in meta["DimSize"]]
    return meta

def mhd_memmap(mhd_path: str):
    meta = parse_mhd(mhd_path)
    X, Y, Z = meta["DimSize"]  # MHD usually stores as X Y Z
    dtype = _MHD_DTYPE[meta["ElementType"]]
    raw_rel = meta["ElementDataFile"]
    raw_path = os.path.join(os.path.dirname(mhd_path), raw_rel)

    msb = str(meta.get("ByteOrderMSB", "False")).lower() == "true"
    if msb:
        dtype = dtype.newbyteorder(">")

    # We map as (Z, Y, X) for convenience in python
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

    return (max(1, min(z, cz)), max(1, min(y, cy)), max(1, min(x, cx)))

def _make_compressor(name: str):
    if name == "zstd":
        return Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    if name == "lz4":
        return Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)
    if name == "none":
        return None
    raise ValueError(f"Unknown compressor: {name}")


# ----------------------------
# Core: build upscaled dask array (streaming-friendly)
# ----------------------------
def build_upscaled_dask_from_mhd(
    mhd_path: str,
    scale: int,
    chunk_mb: int,
):
    mm, meta = mhd_memmap(mhd_path)
    zyx = mm.shape
    dtype = mm.dtype.newbyteorder("=")
    bpp = mm.dtype.itemsize

    spacing_xyz = meta.get("ElementSpacing", None)  # typically [sx, sy, sz] in micrometers (or whatever unit)
    print(f"Source shape (z,y,x): {zyx}, dtype={dtype}, spacing(x,y,z)={spacing_xyz}")

    chunks = choose_chunks(zyx, bpp, target_chunk_mb=chunk_mb)
    print(f"Using input chunks (z,y,x): {chunks} (~{chunk_mb} MB target per chunk)")
    darr = da.from_array(mm, chunks=chunks, asarray=False)

    up = darr
    if scale != 1:
        up = da.repeat(up, scale, axis=0)
        up = da.repeat(up, scale, axis=1)
        up = da.repeat(up, scale, axis=2)

    out_shape = tuple(int(s * scale) for s in zyx)
    print(f"Upscaled shape (z,y,x): {out_shape}")

    return up, meta, chunks, zyx, out_shape, dtype, bpp


# ----------------------------
# Writers
# ----------------------------
def write_zarr_or_omezarr(
    up: da.Array,
    out_dir: str,
    out_chunks,
    compressor,
    meta: dict,
    source_mhd: str,
    scale: int,
    source_shape_zyx,
    output_shape_zyx,
    dtype_str: str,
    fmt: str,
):
    # Clean output if exists
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    # Ensure chunk-size config doesnt try to assemble tiny chunks
    bpp = np.dtype(dtype_str).itemsize if isinstance(dtype_str, str) else np.dtype(dtype_str).itemsize
    out_chunk_bytes = out_chunks[0] * out_chunks[1] * out_chunks[2] * bpp
    dask.config.set({"array.chunk-size": max(out_chunk_bytes, 32 * 1024 * 1024)})

    store = zarr.DirectoryStore(out_dir)
    with ProgressBar():
        up2 = up.rechunk(out_chunks)
        up2.to_zarr(store, overwrite=True, compressor=compressor)

    
    info = {
        "source": os.path.abspath(source_mhd),
        "element_spacing_xyz": meta.get("ElementSpacing", None),
        "scale_factor": scale,
        "source_shape_zyx": list(source_shape_zyx),
        "output_shape_zyx": list(output_shape_zyx),
        "chunks_zyx": list(out_chunks),
        "dtype": str(dtype_str),
        "format": fmt,
    }
    with open(os.path.join(out_dir, ".atlas_upscale_meta.json"), "w") as f:
        json.dump(info, f, indent=2)

    
    if fmt == "ome.zarr":
        _add_ome_zarr_metadata(
            out_dir=out_dir,
            meta=meta,
            scale_factor=scale,
            output_shape_zyx=output_shape_zyx,
        )

    print(f"✅ Finished. {fmt} written to: {out_dir}")

def _add_ome_zarr_metadata(out_dir: str, meta: dict, scale_factor: int, output_shape_zyx):
    """
    Minimal OME-Zarr metadata so the output is OME-compatible.
    We keep the data at root array (path "0") (which is what zarr.to_zarr writes).
    """
    # ElementSpacing usually is [sx, sy, sz] (x,y,z). Our array is (z,y,x).
    spacing_xyz = meta.get("ElementSpacing", None)

    # New voxel spacing after upscaling (physical size per voxel decreases)
    # If spacing_xyz is missing, we still write axes without scale.
    if spacing_xyz and len(spacing_xyz) == 3:
        sx, sy, sz = spacing_xyz  # x,y,z
        new_spacing_zyx = [sz / scale_factor, sy / scale_factor, sx / scale_factor]
        coord_transform = [{"type": "scale", "scale": new_spacing_zyx}]
        axes = [
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ]
    else:
        coord_transform = []
        axes = [
            {"name": "z", "type": "space"},
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"},
        ]

    # Root zattrs for OME-Zarr
    zattrs_path = os.path.join(out_dir, ".zattrs")
    root_attrs = {}
    if os.path.exists(zattrs_path):
        # If something already exists, merge conservatively
        try:
            with open(zattrs_path, "r") as f:
                root_attrs = json.load(f)
        except Exception:
            root_attrs = {}

    root_attrs["multiscales"] = [{
        "version": "0.4",
        "name": "upscaled_labels",
        "axes": axes,
        "datasets": [{
            "path": "0",
            "coordinateTransformations": coord_transform
        }]
    }]

    
    root_attrs.setdefault("image-label", True)

    with open(zattrs_path, "w") as f:
        json.dump(root_attrs, f, indent=2)

def write_tiff_stack(up: da.Array, out_tif: str, max_slices: int = 2000):
    """
    Writes a Z-stack TIFF (pages = z slices).
    This is only realistic for smaller volumes.
    """
    try:
        import tifffile
    except ImportError:
        raise RuntimeError("Missing dependency: tifffile. Install with: pip install tifffile")

    z, y, x = up.shape
    z = int(z)

    if z > max_slices:
        raise RuntimeError(
            f"Refusing to write TIFF stack with {z} slices (too large). "
            f"Use --format ome.zarr or zarr for huge outputs, or lower scale / subset data."
        )

    os.makedirs(os.path.dirname(out_tif) or ".", exist_ok=True)

    print(f"Writing TIFF stack: {out_tif} (slices={z})")
    with tifffile.TiffWriter(out_tif, bigtiff=True) as tw:
        for zi in range(z):
            # compute one slice at a time
            sl = up[zi, :, :].compute()
            tw.write(sl, contiguous=True)
            if (zi + 1) % 50 == 0:
                print(f"  wrote {zi+1}/{z} slices")

    print(f"✅ Finished. TIFF written to: {out_tif}")

def write_nrrd(up: da.Array, out_nrrd: str, max_gb: float = 2.0):
    """
    NRRD typically requires materializing an array (common libs don't stream well).
    We'll guard against huge outputs.
    """
    try:
        import nrrd  # pynrrd
    except ImportError:
        raise RuntimeError("Missing dependency: pynrrd. Install with: pip install pynrrd")

    nbytes = np.prod(up.shape) * np.dtype(up.dtype).itemsize
    gb = nbytes / (1024**3)

    if gb > max_gb:
        raise RuntimeError(
            f"Refusing to write NRRD of ~{gb:.2f} GB (too large for safe local export). "
            f"Use --format ome.zarr or zarr."
        )

    arr = up.compute()  # materialize
    os.makedirs(os.path.dirname(out_nrrd) or ".", exist_ok=True)
    nrrd.write(out_nrrd, arr)
    print(f"✅ Finished. NRRD written to: {out_nrrd}")

def write_nifti(up: da.Array, out_nii: str, max_gb: float = 2.0):
    """
    NIfTI export also typically requires materializing (depending on workflow).
    We'll guard against huge outputs.
    """
    try:
        import nibabel as nib
    except ImportError:
        raise RuntimeError("Missing dependency: nibabel. Install with: pip install nibabel")

    nbytes = np.prod(up.shape) * np.dtype(up.dtype).itemsize
    gb = nbytes / (1024**3)

    if gb > max_gb:
        raise RuntimeError(
            f"Refusing to write NIfTI of ~{gb:.2f} GB (too large for safe local export). "
            f"Use --format ome.zarr or zarr."
        )

    # Our up is (z,y,x). NIfTI convention is usually (x,y,z).
    arr_zyx = up.compute()
    arr_xyz = np.transpose(arr_zyx, (2, 1, 0))

    affine = np.eye(4)  # placeholder (you can incorporate spacing later)
    img = nib.Nifti1Image(arr_xyz, affine)

    os.makedirs(os.path.dirname(out_nii) or ".", exist_ok=True)
    nib.save(img, out_nii)
    print(f"✅ Finished. NIfTI written to: {out_nii}")


# ----------------------------
# Main entry: upscale + write in chosen format
# ----------------------------
def upscale_streaming(
    mhd_path: str,
    output: str,
    fmt: str,
    scale: int = 2,
    chunk_mb: int = 128,
    compressor: str = "zstd",
):
    up, meta, in_chunks, source_shape_zyx, out_shape_zyx, dtype, bpp = build_upscaled_dask_from_mhd(
        mhd_path=mhd_path,
        scale=scale,
        chunk_mb=chunk_mb,
    )

    # Output chunks: keep same as input chunks by default (works well)
    out_chunks = in_chunks
    print(f"Output chunks (z,y,x): {out_chunks}")

    comp = _make_compressor(compressor)

    fmt = fmt.lower()
    if fmt in ("zarr", "ome.zarr"):
        # For zarr and ome.zarr output must be a directory
        write_zarr_or_omezarr(
            up=up,
            out_dir=output,
            out_chunks=out_chunks,
            compressor=comp,
            meta=meta,
            source_mhd=mhd_path,
            scale=scale,
            source_shape_zyx=source_shape_zyx,
            output_shape_zyx=out_shape_zyx,
            dtype_str=str(dtype),
            fmt=fmt,
        )
        return

    
    if fmt == "tiff":
        write_tiff_stack(up=up, out_tif=output)
        return

    if fmt == "nrrd":
        write_nrrd(up=up, out_nrrd=output)
        return

    if fmt in ("nifti", "nii", "nii.gz"):
        write_nifti(up=up, out_nii=output)
        return

    raise ValueError(f"Unknown format: {fmt}")


def main():
    p = argparse.ArgumentParser(
        description="Streaming label upscaler (MHD/RAW -> Zarr/OME-Zarr/other exports) with Dask."
    )
    p.add_argument("--input", required=True, help="Path to annotation .mhd")
    p.add_argument("--output", required=True, help="Output path (dir for zarr/ome.zarr; file for tiff/nrrd/nifti)")
    p.add_argument("--format", default="zarr",
                   choices=["zarr", "ome.zarr", "tiff", "nrrd", "nifti"],
                   help="Output format (default zarr). Use ome.zarr for OME metadata.")
    p.add_argument("--scale", type=int, default=2, help="Integer scale factor (default 2)")
    p.add_argument("--chunk-mb", type=int, default=128, help="Target chunk size in MB (default 128)")
    p.add_argument("--compressor", default="zstd", choices=["zstd", "lz4", "none"], help="Zarr compressor")
    args = p.parse_args()

    upscale_streaming(
        mhd_path=args.input,
        output=args.output,
        fmt=args.format,
        scale=args.scale,
        chunk_mb=args.chunk_mb,
        compressor=args.compressor,
    )

if __name__ == "__main__":
    main()