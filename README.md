# 3D-annotation-image-upscaler-for-huge-images
## code snippet to upscale an 3D annotation file to TB size independed of working memory (PROOF-OF-CONCEPT)

Update 8/2023: there are plans to further develop this idea into a full-fledged tool for use with omero (https://www.openmicroscopy.org/). stay tuned

Even today there are studies in which the regions are marked manually instead of using the significantly more objective alternative of atlas registration with subsequent upscaling. The Python code presented here can upscale an annotation image to a high resolution image without loss of annotation labels. **Note:** Since this is a common problem in neuroscience, it is unlikely that no one has solved it yet. 

The code is mostly based on the "memmap" function of numpy and it upscales in 2 dimensions per step using the [2D resize function of the Pillow package](https://pillow.readthedocs.io/en/stable/reference/Image.html) and rotates the image virtually after the steps to upscale the whole 3D image. The saving is done by the Tiffwriter of the Tifffile package. This makes it possible to upscale an image to an almost unlimited size. For example, on a PC with 8 GB RAM, a 300 MB image can be successfully upscaled to a 450 GB image in about 3 hours (about 43 MB write/s).

<p align="center">
<img src="https://github.com/SaibotMagd/3D-annotation-image-upscaler-for-huge-images/blob/main/3D-AIUdocs/src_image_example1.png" width="300">
<img src="https://github.com/SaibotMagd/3D-annotation-image-upscaler-for-huge-images/blob/main/3D-AIUdocs/src_image_hist_example1.png" width="300">
</p>
<p align="center">
<img src="https://github.com/SaibotMagd/3D-annotation-image-upscaler-for-huge-images/blob/main/3D-AIUdocs/tar_image_example1.png" width="300">
<img src="https://github.com/SaibotMagd/3D-annotation-image-upscaler-for-huge-images/blob/main/3D-AIUdocs/tar_image_hist_example1.png" width="300">
</p>

## features:

- upscale an 3D annotation file pixel per pixel without interpolation (target size nearly infinite and independent of working memory)
- use the shape of an target image for matching (default) or set it manually


## dependencies:

  - datetime, numpy, os, tifffile, PIL
  - needs 1.5-times the target size free space on harddrive for processing (i.e. upscaling to target of 450GB needs about 950GB free space (450GB for the result file, 500GB for processing) 
  
## Why is it important?:

After registering a huge lightsheet brain dataset from a mouse onto the allen mouse brain atlas ([Wang et al. 2020 Cell](https://doi.org/10.1016/j.cell.2020.04.007)) or vice versa, it could be important to mark a specific or all regions from the atlas onto the tissue in the highest resolution. To perform the registration its necessary to downscale the image (to use the full resolution image does't increase the registration quality since the atlas is currently only offered in 10um resolution; also the performance would be a nightmare). To visualize segmented vessels or marked cells (i.e. using [Kirst et al. 2020 Cell](https://doi.org/10.1016/j.cell.2020.01.028)) onto a region defined by the atlas annotation in full resolution an upscaling of the registered annotation file is necessary. ***Important to know: you have to disable any kind of interpolation algorithm, as it makes the annotations unusable.***

The result could look similar to this ([Blue Brain Cell Atlas](https://bbp.epfl.ch/nexus/cell-atlas/)):

<p align="center">
<a href="https://bbp.epfl.ch/nexus/cell-atlas/">
<img src="https://github.com/SaibotMagd/3D-annotation-image-upscaler-for-huge-images/blob/main/3D-AIUdocs/blue_brain_cell_atlas_example1.png" 
 alt="Blue Brain Cell Atlas Example" width="300" hspace="40"/></a>

Today there's no function to do this in neither Imaris nor Arivis (very high-quality proprietary software systems for the display and processing of particularly large neuroscientific image data), even if they are able to use an upscaled annotation file as an overlay, they cannot create these overlays themselves. In ImageJ its possible to do the upscaling with the *scale function* but it uses the working memory, so large images are impossible and also the performance seems pretty bad. [Convert3D from the authors of itk-snap](http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.C3D) can upscale images slices wise and disable the interpolation, but it can only upscale images with less then 256 colors and is incapable to upscale images larger then some GB. 

## How to help?

- tell me about a way to do this more efficient; how do others solve this task?

## Next steps?:

- create a CLI for easier use
- select one or a bunch of labels to create an overlay mask for visualizing or to as ROI
- parallelizing the algorithm (depending on the speed of the disc drive)
- use N5 or HDF5 to save the data for better performance and data storage
- combine the upscaler and the cell plotter (work in progress)
- integrate the upscaler into Arivis to plot the cells depending on a specific region
