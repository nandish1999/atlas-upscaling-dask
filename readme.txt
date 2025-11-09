conda activate atlas
cd /Users/acharnandish/3D-annotation-image-upscaler-for-huge-images-main
python upscale_dask.py


python upscale.py --input p4791e-ext-d000030_3Drecon-ADMBA-P56_pub/3Drecon-ADMBA-P56_annotation.mhd --output atlas_upscaled_v4.zarr --scale 2



conda activate atlas
/usr/bin/time -l python upscale.py \
  --input p4791e-ext-d000030_3Drecon-ADMBA-P56_pub/3Drecon-ADMBA-P56_annotation.mhd \
  --output atlas_upscaled_bench.zarr --scale 2