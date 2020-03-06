#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:12:13 2020

@author: claytonwandishin
"""


import numpy as np
import os
from tifffile import imread, imsave, TiffFile
from sklearn import cluster
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, fixed
from skimage.data import *
import cv2 as cv
from scipy.ndimage.morphology import distance_transform_edt
import itk
from skimage.morphology import remove_small_objects, dilation, disk
from skimage.filters import threshold_otsu, sobel
from skimage.measure import label
from aicssegmentation.core.visual import blob2dExplorer_single, fila2dExplorer_single, random_colormap
from aicssegmentation.core.seg_dot import dot_2d, logSlice
from aicssegmentation.core.vessel import filament_2d_wrapper
from aicssegmentation.core.utils import hole_filling
from aicssegmentation.core.pre_processing_utils import intensity_normalization, suggest_normalization_param, edge_preserving_smoothing_3d
from skimage import data, io
from skimage.color import rgb2gray
from aicsimageio import AICSImage
from aicssegmentation.cli.to_analysis import simple_builder, masked_builder

def km_clust(array, n_clusters):
    
    # Create a line array, the lazy way
    X = array.reshape((-1, 1))
    # Define the k-means clustering problem
    k_m = cluster.KMeans(n_clusters=n_clusters, n_init=4)
    # Solve the k-means clustering problem
    k_m.fit(X)
    # Get the coordinates of the clusters centres as a 1D array
    values = k_m.cluster_centers_.squeeze()
    # Get the label of each point
    labels = k_m.labels_
    return(values, labels)

#change this directory if needed (mask)
fn = '/Users/claytonwandishin/Downloads/161124u2os#5ncs6NCS_0_1c1-0136.tif'
os.path.isfile(fn)

tif = TiffFile(fn)

num_img = len(tif.pages)
print(num_img)

img_seq = []
for ii in range(num_img):
    img_seq.append(imread(fn, key=ii))
img = img_seq[0]
#plt.imshow(img)


#suggest_normalization_param(img)
#img_norm = intensity_normalization(img, [0.5, 8.0])

#smooths the edges
img_smooth = edge_preserving_smoothing_3d(img)

#identifies objects
interact(blob2dExplorer_single, im=fixed(img_smooth), \
         sigma=widgets.FloatRangeSlider(value=(1,5), min=1, max=15,step=1,continuous_update=False),  \
         th=widgets.FloatSlider(value=0.02,min=0.01, max=0.1, step=0.01,continuous_update=False));

#plt.imshow(img_smooth)

#ksize = (60,60)
#blur_smooth = cv2.blur(np.float32(img_smooth), ksize)
#nuc_detection = dot_2d(img_smooth, 11)

#creates a mask
nuc_mask = img_smooth>0.0015

#fills small holes
final_seg = hole_filling(remove_small_objects(nuc_mask),1, 1500)

#inverts the image and crops it
inv_seg = np.invert(final_seg)
cropped = inv_seg[0:1983,0:1983]
#fills holes in cropped image
cropped = hole_filling(remove_small_objects(cropped),1, 1600)

plt.imshow(cropped, cmap='gray')
#plt.imshow(circle)


#change this directory if needed (actual image)
fn = '/Users/claytonwandishin/UNC_PCNA/UNC_PCNA_Split/161124u2os#5ncs6NCS_0_1c1-0136.tif'
os.path.isfile(fn)

tif1 = TiffFile(fn)

num_img_1 = len(tif1.pages)
print(num_img_1)

#applies the previously generated mask to the image but first treats it
img_seq1 = []
for ii in range(num_img):
    img_seq1.append(imread(fn, key=ii))
img1 = img_seq1[0]
img2 = img1[0:1983,0:1983]

#this is the most important line as it applies the mask to the image
masked_image = img2 * cropped
nuc_label = dilation(label(masked_image), disk(3))

#plt.imshow(nuc_label, cmap='gray')

#this shows the image after being treated with the mask
plt.imshow(masked_image, cmap="gray")

#this next block of code takes the kmeans clustering to identify puncta
# Read the data as greyscale 
img = imread(fn)
#img = cv2.imread('Timeseries2.tif')
# Group similar grey levels using 8 clusters
values, labels = km_clust(img, n_clusters = 3)
# Create the segmented array from labels and values
img_segm = np.choose(labels, values)
# Reshape the array as the original image
img_segm.shape = img.shape
# Get the values of min and max intensity in the original image
vmin = img.min()
vmax = img.max()
fig = plt.figure(1)

'''
# Plot the original image
ax1 = fig.add_subplot(1,2,1)
ax1.imshow(img,cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
ax1.set_title('Original image')
# Plot the simplified color image
ax2 = fig.add_subplot(1,2,2)
ax2.imshow(img_segm, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
ax2.set_title('Simplified levels')
# Get rid of the tick labels
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax2.set_xticklabels([])
ax2.set_yticklabels([])
plt.show()
'''
plt.figure(1)
plt.imshow(img_segm,cmap="gray")
plt.show()
