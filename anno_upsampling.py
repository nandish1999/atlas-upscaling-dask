#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 12:40:24 2022

@author: saibotMagd

"""

if __name__ == "__main__":  
   
  import datetime
  import numpy as np
  import os  
  import tifffile
  from tifffile import TiffWriter
  from PIL import Image
  import os

  def cls():
    os.system('cls' if os.name=='nt' else 'clear')

  
  #%%############################################################################
  ### Initialization - set parameters
  ###############################################################################

  directory = '/scratch/201205_LSstandardMouse/'
  anno_file = 'bspline_annotation_result.1.tif'
  
  # define the source resolution out of the source annotation file
  source_file = os.path.join(directory, anno_file)
  source_mm = tifffile.imread(source_file)
  src_x, src_y, src_z = source_mm.shape[2], \
                        source_mm.shape[1], \
                        source_mm.shape[0]
  
  # define the resolution for the target file either manuel or 
  # match a target file so the annotation can be used as overlay                        
  target_shape_file = ""
  
  if len(target_shape_file) != 0: 
    memmap_image        = tifffile.memmap(os.path.join(directory, target_shape_file))
    tar_x, tar_y, tar_z = memmap_image.shape[2], memmap_image.shape[1], memmap_image.shape[0]
    del memmap_image, target_shape_file
  else:
    tar_x, tar_y, tar_z = src_x*2, src_y*2, src_z*2 #matching resolution
    
  ###############################################################################
  # ##  ### 1. STEP: upscale x,y direction and hold z stable
  ###############################################################################
  cls()
  starttime = datetime.datetime.now()
  # create the final upscale file (just reserves the space on HDD)
  target_file = os.path.join(directory, "anno_upscaled_to_stitched.npy")
  target_mm = np.memmap(target_file, dtype="float32", mode="w+", shape=(259,tar_y,tar_x))
  
  # loop over all slices:
  looptime = 0
  for i in range(source_mm.shape[0]):
    looptime += datetime.datetime.now()
  
    slice = np.array(Image.fromarray(source_mm[i]).resize((tar_x, tar_y), Image.NEAREST))
    target_mm[i] = slice
    target_mm.flush()
    if i!=0 and (round(i/source_mm.shape[0],2)*100)%10 == 0: 
      print(f"slice {i} of {len(source_mm[:,:,0])} ({round(i*100/source_mm.shape[0],1)}%) in {datetime.datetime.now()-looptime}")
  
  print("##########################################")
  print(f"S1 x,y upscale done: {len(source_mm[:,:,0])} slices upscaled in {datetime.datetime.now()-starttime}")
  print("##########################################")

  ###############################################################################
  # ##  ### 2. STEP: upscale z direction and hold xy stable
  ###############################################################################
  # # load the source registration file as mm (write protected) [z,y,x]
  source_file = os.path.join(directory, "anno_upscaled_to_stitched.npy")
  source_mm = np.memmap(source_file, dtype="float32", mode="r", shape=(src_z,tar_y,tar_x))
  
  # create the final upscale file (just reserves the space on HDD)
  target_file = os.path.join(directory, "anno_upscaled_to_rot-stitched.npy")
  target_rot = np.memmap(target_file, dtype="float32", mode="w+", shape=(tar_y,tar_z,tar_x))
  
  source_mm = np.rot90(source_mm, 1, axes=(0,1))
  
  starttime2 = datetime.datetime.now()
  
  looptime = 0
  # loop over all slices:
  for i in range(source_mm.shape[0]):
    looptime += datetime.datetime.now()
    
    slice = np.array(Image.fromarray(source_mm[i]).resize((tar_x,tar_z), Image.NEAREST))
    target_rot[i] = slice
    target_rot.flush()
    
    if i!=0 and (round(i*100/source_mm.shape[0]))%20 == 0: 
     print(f"slice {i} of {len(source_mm[:,:,0])} ({round(i*100/source_mm.shape[0],1)}%) in {datetime.datetime.now()-looptime}")
  
  target_rot = np.rot90(target_rot, -1, axes=(0,1))
  target_rot.flush()
  
  print("##########################################")
  print(f"{len(source_mm[:,:,0])} slices upscaled in {datetime.datetime.now()-starttime2}")
  print("##########################################")
  ###############################################################################
  # ##  ### 3. STEP: convert to tif
  ###############################################################################
  

  # writing
  with TiffWriter(os.path.join(directory, "bspline_annotation_upscaled_v3.tif"), bigtiff=True) as tif:
      looptime = 0
      for i in range(target_rot.shape[0]):
        looptime += datetime.datetime.now()  
        tif.save(target_rot[i], photometric='minisblack') # min-is-black
        if i!=0 and (round(i*100/source_mm.shape[0]))%20 == 0: 
          print(f"slice {i} of {target_rot.shape[0]} ({round(i*100/target_rot.shape[0],1)}%) in {datetime.datetime.now()-looptime}")
  
  ###############################################################################
  # ##  ### 4. STEP: delete the working files
  ###############################################################################
  os.remove(target_file)
  os.remove(source_file)
  
  print("##########################################")
  print(f"Full processing time: {datetime.datetime.now()-starttime}")
  print("##########################################")
