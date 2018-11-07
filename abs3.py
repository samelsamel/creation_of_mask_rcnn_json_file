import simplejson as json
from glob import glob
import pickle
import sys
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
import scipy
import cv2
import numpy as np
import numba 
from tqdm import tqdm
from PIL import Image
from pycocotools.mask import area, encode, toBbox
import itertools 
import os
from skimage.io import imread
import pdb

import pandas as pd
from skimage.io import imsave

# In the dataset, their is an image folder and a mask folder  and the annotation is the json file that will be created.  


IMAGE_TRAIN_FOLDER = 'images/*'
MASK_TRAIN_FOLDER = 'masks/*'
ANNOTATIONS = 'full_last_mask.json'





def create_json(image_loc, mask_loc,i):
    #every dict has a filename, size and region part which will be extented afterwads with (shape_attributes, all_points_x and all_points_y)
    dic = { 'filename' : [],
             'regions': {
    },
             'size': []
    }
    
    x = []
    y = []
    image = sorted(os.listdir('full path to your image directory'))[i]
    mask = sorted(os.listdir('full path to your mask directory'))[i]
    print(image)
    print(mask)
    print(i)
    dic['size'] = 2560000   #give the size of the image 
    dic['filename']= image
    y_true = imread('path to mask directory '+mask, dtype =np.uint8) # y_true is the real mask 

    bw = closing(y_true < 1, square(3)) 
    # label image regions
    label_image = label(bw)
    j=0
    #here we extract region propasals for every image
    for region in regionprops(label_image):
        img_0 = np.zeros(label_image.shape, dtype =np.uint8)
        img = np.zeros(label_image.shape, dtype =np.uint8, order='F')
        img[label_image==j+1] = 255
        img_0[label_image==j+1] = 255
        rles = encode(img)
        areas = area(rles)
        if areas < 0.0:            
            pdb.set_trace()
        bboxs = toBbox(rles).tolist()
        mask_new, contours, hierarchy = cv2.findContours(
        img_0.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        k = 0

        for contour in contours:
            k = k + 1 
            #for every contour in contours we create a dict to add the regions of an image
            dict_2 = {
             'region_attributes': {},
             'shape_attributes': {
             'all_points_x': [],
             'all_points_y': [],
             'name': 'polygon'}}
            x = list(contour[:,0,0])
            y = list(contour[:,0,1])
        dict_2['shape_attributes']['all_points_x']=x
        dict_2['shape_attributes']['all_points_y']=y
        dic['regions'][str(j)]=dict_2
        j = j + 1
        
    return dic
        


#l is the final dict and i is the image_id 
i = 0
l = {}
#we create a dictionary for every image 
# in this case for example we have 200 images, just change it to your number of images
while (i < 200):
    dic = create_json(IMAGE_TRAIN_FOLDER, MASK_TRAIN_FOLDER,i)
    l[str(i)]=dic
    i = i+ 1
df = pd.DataFrame(l)
df.to_json(path_or_buf=ANNOTATIONS)
