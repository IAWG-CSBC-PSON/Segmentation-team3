#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 21:07:13 2020

@author: claytonwandishin
"""
import numpy as np
import os
from tifffile import imread
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2
from skimage import measure
from aicssegmentation.core.pre_processing_utils import intensity_normalization, suggest_normalization_param, edge_preserving_smoothing_3d
from skimage.filters import sobel, threshold_yen
import csv

#defines a circular mask function
def cmask(radius):
    kernel = np.zeros((2 * radius + 1, 2 * radius + 1), np.uint8)
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask = x ** 2 + y ** 2 <= radius ** 2
    kernel[mask] = 1
    return kernel

#This needs to be the path to the folder containing the tif files
DATAPATH = "/home/acc/hackathon-data/Segmentation/Vanderbilt_live_cell"
#DATAPATH = "/Users/claytonwandishin/UNC_PCNA/Vandy_Live_Cell"

#creates a list from all the files in the folder
image_list = []
for root, dirs, files in os.walk(DATAPATH):
    for file in files:
        if file.endswith('.tif'):
            image_list.append(file)
            
#creates an empty list for the object counts
object_counts = []    
     
#loops through every image in the image_list
for x in image_list:
    im_path = os.path.join(DATAPATH,x)
    os.path.isfile(im_path)
    img = imread(im_path)
    
    #this could definitely be a more complicated normalization function
    #suggest_normalization_param(img)
    img_norm = intensity_normalization(img, [5.0, 15.0])
    img_smooth = edge_preserving_smoothing_3d(img_norm)
    bc_img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, cmask(8))
    bc_img = sobel(bc_img)
    
    #thresholds the image based on yen's method
    block_size = 35
    thresh = threshold_yen(bc_img)
    binary = bc_img > thresh
    
    #helps to subtract the background
    n = 12
    l = 256
    np.random.seed(1)
    im = np.zeros((l, l))
    points = l * np.random.random((2, n ** 2))
    im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
    blobs = binary > 0.7 * binary.mean()
    
    #labels the individual objects
    all_labels = measure.label(blobs)
    blobs_labels = measure.label(blobs, background=0)
        
    #blurs the nuclei to remove small objects and then creates an ndimage so that the connected components can be counted
    blur_radius = 4.0
    blur_blob = ndimage.gaussian_filter(blobs_labels, blur_radius)
    blobs_objects, num_features = ndimage.label(blur_blob)
    
    #prints the count of objects
    #print("Number of objects is {}".format(num_features))
    
    #write each blobs_objects array to a .tif or.png file
    plt.imsave(os.path.join("seg_output",x), blobs_objects, format='png')
    
    #appends the object count to the object_counts list
    object_counts.append(num_features)
    
    #plots the segmented image using an attractive cmap
    #plt.imshow(blobs_labels, cmap='nipy_spectral')

#creates a dictionary from the image_list and object_counts
count_table = dict(zip(image_list, object_counts))

#writes the count_table to a csv file
w = csv.writer(open("count_table.csv", "w"))
for key, val in count_table.items():
    w.writerow([key, val])
