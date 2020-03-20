# Abdur. R. Fayjie, R. Azad, Claude Kauffman, Ismail Ben Ayed, Marco Pedersoli and Jose Dolz "Semi-supervised Few-Shot Learning for Medical Image Segmentation", arXiv preprint arXiv, 2020
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import random 
import pickle
import cv2
import copy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")


###
## Generate Train and Test classes
def Get_tr_te_lists(FSS_add, t_l_path):
    text_file = open(t_l_path, "r")
    Test_list = [x.strip() for x in text_file] 
    Class_list = os.listdir(FSS_add)
    Train_list = []
    for idx in range(len(Class_list)):
        if not(Class_list[idx] in Test_list):
           Train_list.append(Class_list[idx])
    
    return Train_list, Test_list
    
def get_episode_FSS(setX, n_way = 5, k_shot = 1, data_path='./', h=224, w=224):
    indx_c = random.sample(range(0, len(setX)), n_way)
    indx_s = random.sample(range(1, 11), 10)

    support = np.zeros([n_way, k_shot, h,  w, 3], dtype = np.float32)
    smasks  = np.zeros([n_way, k_shot, 56, 56,1], dtype = np.float32)
    query   = np.zeros([n_way,         h,  w, 3], dtype = np.float32)      
    qmask   = np.zeros([n_way,         h,  w, 1], dtype = np.float32)  
                
    for idx in range(len(indx_c)):
        for idy in range(k_shot): # For support set 
            s_img = cv2.imread(data_path + setX[indx_c[idx]] + '/' + str(indx_s[idy]) + '.jpg' )
            s_msk = cv2.imread(data_path + setX[indx_c[idx]] + '/' + str(indx_s[idy]) + '.png' )
            s_img = cv2.resize(s_img,(h, w))
            s_msk = cv2.resize(s_msk,(56,        56))        
            s_msk = s_msk /255.
            s_msk = np.where(s_msk > 0.5, 1., 0.)
            support[idx, idy] = s_img
            smasks[idx, idy]  = s_msk[:, :, 0:1] 
        for idyx in range(1): # For query set consider 1 sample per class
            q_img = cv2.imread(data_path + setX[indx_c[idx]] + '/' + str(indx_s[idyx+k_shot]) + '.jpg' )
            q_msk = cv2.imread(data_path + setX[indx_c[idx]] + '/' + str(indx_s[idyx+k_shot]) + '.png' )
            q_img = cv2.resize(q_img,(h, w))
            q_msk = cv2.resize(q_msk,(h, w))        
            q_msk = q_msk /255.
            q_msk = np.where(q_msk > 0.5, 1., 0.)
            query[idx] = q_img
            qmask[idx] = q_msk[:, :, 0:1]        

    support = support /255.
    query   = query   /255.
       
    return support, smasks, query, qmask    


def get_episode_surrogate(IMG, img_h = 224, img_w = 224, n_way = 1):
  
    Images = np.zeros([n_way, img_h, img_w, 3], dtype = np.float32)
    indx_s = random.sample(range(0, len(IMG)-1), n_way)
    for idy in range(n_way):
        Images[idy]  = IMG[indx_s[idy]]
    Images = Images /255.
       
    return Images

def add_noise(X, X2):
    noisy_data  = np.zeros((X.shape[0]+X2.shape[0], X.shape[1],X.shape[2],X.shape[3]))
    target_data = np.zeros((X.shape[0]+X2.shape[0], int(X.shape[1]/4) ,int(X.shape[2]/4), X.shape[3]))
    noise       = np.random.normal(0, 1, noisy_data.shape)
    random_noise_value = round(random.uniform(0.1, 0.6),3)  
    noisy_data[0:X.shape[0]]  = X  + random_noise_value * noise[0:X.shape[0]] 
    noisy_data[X.shape[0]]    = X2 + random_noise_value * noise[X.shape[0]]
        
    for idx in range(X.shape[0]):
        target_data[idx] = cv2.resize(X[idx],(56, 56))
    target_data[X.shape[0]] = cv2.resize(X2[0],(56, 56))    
    return noisy_data, target_data
    
def get_episode_test(IMG, MSK, img_h = 224, img_w = 224, n_way = 1, k_shot = 1, t_shot = 1):
  
    support = np.zeros([n_way, k_shot, img_h, img_w, 3], dtype = np.float32)
    query   = np.zeros([n_way, img_h, img_w, 3], dtype = np.float32)      
    smask   = np.zeros([n_way, k_shot, int(img_h/4), int(img_w/4)], dtype = np.float32)             
    qmask   = np.zeros([n_way, img_h, img_w], dtype = np.float32)
        
    for idx in range((n_way)):
        indx_s = random.sample(range(0, len(IMG)-1), k_shot+t_shot)
        for idy in range(k_shot): # For support set 
            s_img = IMG[indx_s[idy]]
            s_msk = MSK[indx_s[idy]]
            s_msk = np.array(s_msk, dtype= 'uint8')
            s_msk = cv2.resize(s_msk,(56, 56))  
            s_msk = s_msk /255.
            s_msk = np.where(s_msk > 0., 1., 0.)
            support[idx, idy, :, :, :] = s_img          
            smask[idx, idy]   = s_msk
            
        for idy in range(t_shot): # For query set consider 1 sample per class
            q_img = IMG[indx_s[idy+k_shot]]
            q_msk = MSK[indx_s[idy+k_shot]]
            query[idx, :, :, :] = q_img
            qmask[idx] = q_msk    

    support = support /255.
    query   = query   /255.
    qmask   = qmask   /255.
    qmask   = np.where(qmask > 0., 1., 0.)    
    qmask   = np.expand_dims(qmask, axis=3)
    smask   = np.expand_dims(smask, axis=4)
       
    return support, smask, query, qmask   

def compute_dice(y_pred, y_true, T= 0.5):
    Dice_score = 0
    y_pred = y_pred.reshape(y_pred.shape[0]*y_pred.shape[1]*y_pred.shape[2]*y_pred.shape[3], 1)
    y_true = y_true.reshape(y_true.shape[0]*y_true.shape[1]*y_true.shape[2]*y_true.shape[3], 1)
    y_pred = np.where(y_pred> T, 1., 0)
    y_true = np.where(y_true> 0.5, 1., 0)
    # In binary case F1 is equall to Dice Score (https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2)
    Dice_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)

    return Dice_score
    
        