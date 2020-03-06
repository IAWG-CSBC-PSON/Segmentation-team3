#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:26:29 2020

@author: claytonwandishin
"""

import numpy as np
import os
from tifffile import imread, imsave, TiffFile

import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
from IPython.display import display

import cv2
from scipy.ndimage.morphology import distance_transform_edt
import itk
from skimage.morphology import remove_small_objects, dilation, disk
from skimage.filters import threshold_otsu
from skimage.measure import label
from aicssegmentation.core.visual import blob2dExplorer_single, fila2dExplorer_single, random_colormap
from aicssegmentation.core.seg_dot import dot_2d, logSlice
from aicssegmentation.core.vessel import filament_2d_wrapper
from aicssegmentation.core.utils import hole_filling
from aicssegmentation.core.pre_processing_utils import intensity_normalization, suggest_normalization_param, edge_preserving_smoothing_3d


fn = '/Users/claytonwandishin/UNC_PCNA/161124 u2os #5 ncs 6NCS_0_1c1.tif'
os.path.isfile(fn)

tif = TiffFile(fn)

num_img = len(tif.pages)
print(num_img)

img_seq = []
for ii in range(num_img):
    img_seq.append(imread(fn, key=ii))
img = img_seq[135]
suggest_normalization_param(img)
img_norm = intensity_normalization(img, [0.5, 8.0])
img_smooth = edge_preserving_smoothing_3d(img_norm)
interact(blob2dExplorer_single, im=fixed(img_smooth), \
         sigma=widgets.FloatRangeSlider(value=(1,5), min=1, max=15,step=1,continuous_update=False),  \
         th=widgets.FloatSlider(value=0.02,min=0.01, max=0.1, step=0.01,continuous_update=False));
ksize = (65,65)
blur_smooth = cv2.blur(np.float32(img_smooth), ksize)
nuc_detection = dot_2d(blur_smooth, 11)
nuc_mask = nuc_detection>0.0015
final_seg = hole_filling(remove_small_objects(nuc_mask),.5, 2500)
plt.imshow(final_seg)


