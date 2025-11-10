
import argparse, os, re, json
import numpy as np
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
import zarr
from numcodecs import Blosc



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
    
    if "DimSize" not in meta or "ElementDataFile" not in meta or "ElementType" not in meta:
        raise ValueError("MHD missing required fields (DimSize, ElementType, ElementDataFile).")
    
    meta["DimSize"] = [int(x) for x in meta["DimSize"]]
    return meta

def mhd_memmap(mhd_path):
    
    meta = parse_mhd(mhd_path)
    X, Y, Z = meta["DimSize"]  
    dtype = _MHD_DTYPE[meta["ElementType"]]
    raw_rel = meta["ElementDataFile"]
    raw_path = os.path.join(os.path.dirname(mhd_path), raw_rel)

    
    msb = str(meta.get("ByteOrderMSB", "False")).lower() == "true"
    if msb:
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
    
    return (max(1, min(z, cz)), max(1, min(y, cy)), max(1, min(x, cx)))



def upscale_streaming_mhd_to_zarr(mhd_path, out_zarr, scale=2, chunk_mb=128, compressor="zstd"):
    
    mm, meta = mhd_memmap(mhd_path)          
    zyx = mm.shape
    dtype = mm.dtype.newbyteorder("=")
    bpp = mm.dtype.itemsize

    
    print(f"Source shape (z,y,x): {zyx}, dtype={dtype}, spacing={meta.get('ElementSpacing', 'NA')}")

 
    chunks = choose_chunks(zyx, bpp, target_chunk_mb=chunk_mb)
    print(f"Using input chunks (z,y,x): {chunks}  (~{chunk_mb} MB target per chunk)")
    darr = da.from_array(mm, chunks=chunks, asarray=False)


    up = darr
    if scale != 1:
        up = da.repeat(up, scale, axis=0)
        up = da.repeat(up, scale, axis=1)
        up = da.repeat(up, scale, axis=2)
    out_shape = tuple(int(s * scale) for s in zyx)
    print(f"Upscaled shape (z,y,x): {out_shape}")

   
    if compressor == "zstd":
        comp = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    elif compressor == "lz4":
        comp = Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)
    else:
        comp = None

    
    out_chunks = chunks
    print(f"Output chunks (z,y,x): {out_chunks}")

    
    out_chunk_bytes = out_chunks[0]*out_chunks[1]*out_chunks[2]*bpp
    dask.config.set({"array.chunk-size": max(out_chunk_bytes, 32*1024*1024)})  # at least 32MB

   
    if os.path.exists(out_zarr):
        
        import shutil
        shutil.rmtree(out_zarr)

    store = zarr.DirectoryStore(out_zarr)
    with ProgressBar():
        up = up.rechunk(out_chunks)
        up.to_zarr(store, overwrite=True, compressor=comp)

    
    info = {
        "source": os.path.abspath(mhd_path),
        "element_spacing_um": meta.get("ElementSpacing", None),
        "scale_factor": scale,
        "source_shape_zyx": zyx,
        "output_shape_zyx": out_shape,
        "chunks_zyx": out_chunks,
        "dtype": str(dtype),
    }
    with open(os.path.join(out_zarr, ".atlas_upscale_meta.json"), "w") as f:
        json.dump(info, f, indent=2)
    print("✅ Finished. Zarr written to:", out_zarr)



def main():
    p = argparse.ArgumentParser(description="Streaming label upscaler (MHD/RAW -> Zarr) with Dask.")
    p.add_argument("--input", required=True, help="Path to annotation .mhd")
    p.add_argument("--output", required=True, help="Output .zarr directory")
    p.add_argument("--scale", type=int, default=2, help="Integer scale factor (default 2)")
    p.add_argument("--chunk-mb", type=int, default=128, help="Target chunk size in MB (default 128)")
    p.add_argument("--compressor", default="zstd", choices=["zstd","lz4","none"], help="Zarr compressor")
    args = p.parse_args()

    upscale_streaming_mhd_to_zarr(
        args.input, args.output, scale=args.scale, chunk_mb=args.chunk_mb, compressor=args.compressor
    )

if __name__ == "__main__":
    main()
