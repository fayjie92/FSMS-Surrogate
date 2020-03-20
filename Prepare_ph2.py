# Abdur. R. Fayjie, R. Azad, Claude Kauffman, Ismail Ben Ayed, Marco Pedersoli and Jose Dolz "Semi-supervised Few-Shot Learning for Medical Image Segmentation", arXiv preprint arXiv, 2020
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 21:56:46 2020

@author: Reza Winchester
"""
import numpy as np
from os import path
import pickle
import glob
import scipy.io as sio
import scipy.misc as sc

height, width, channels = 224, 224, 3


Dataset_add = '/PH2Dataset/'

bimg_add = '_Dermoscopic_Image'
bseg_add = '_lesion'


Data_list = glob.glob(Dataset_add+'*')
# check availability of the dataset
Images  = []
Masks   = []

print('Reading data')
for idx in range(len(Data_list)):
    print(idx+1)
    img_add = Data_list[idx]
    img_add2 = Dataset_add+ img_add[len(img_add)-6: len(img_add)]+'/'+img_add[len(img_add)-6: len(img_add)] + bimg_add+'/'+img_add[len(img_add)-6: len(img_add)]+'.bmp'
    msk_add2 = Dataset_add+ img_add[len(img_add)-6: len(img_add)]+'/'+img_add[len(img_add)-6: len(img_add)] + bseg_add+'/'+img_add[len(img_add)-6: len(img_add)]+'_lesion.bmp'
    img = sc.imread(img_add2)
    msk = sc.imread(msk_add2)   
    img = np.double(sc.imresize(img, [height, width, channels], interp='bilinear', mode = 'RGB'))
    msk = np.double(sc.imresize(msk, [height, width], interp='bilinear'))
    Images.append(img)
    Masks.append(msk)
  
    
Images = np.array(Images)
Masks  = np.array(Masks)
print(Images.shape)
print(Masks.shape)

np.save('data_test_ph2', Images) 
np.save('mask_test_ph2', Masks)      
