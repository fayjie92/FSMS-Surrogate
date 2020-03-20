# Abdur. R. Fayjie, R. Azad, Claude Kauffman, Ismail Ben Ayed, Marco Pedersoli and Jose Dolz "Semi-supervised Few-Shot Learning for Medical Image Segmentation", arXiv preprint arXiv, 2020
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
import Model as  M
import matplotlib.pyplot as plt
import Utilz as U
import numpy as np
import pickle 
import random
import cv2

## Get options
Best_performance = 0
epochs           = 30
tr_iterations    = 1000
it_eval          = 200
LR               = 0.0001
KSHOT            = 1
NWAY             = 5
Valid_Dice       = []

## Load FSS, ISIC and ph2 datasets  
Surrogate_set_ISIC = np.load('data_train_isic.npy')
ph2_img = np.load('data_test_ph2.npy')
ph2_msk = np.load('mask_test_ph2.npy')    
t_l_path   = './fss_test_set.txt'     
FSS_add    = './fss_dataset/'
Train_list_FSS, Surrogate_set_FSS = U.Get_tr_te_lists(FSS_add, t_l_path)   
print('All datasets are loaded')
                 
# Build the model
model, Surrogate_model = M.my_model(encoder = 'VGG', input_size = (224, 224, 3), k_shot =KSHOT, learning_rate = LR, learning_rate2 = LR)
model.summary()
Surrogate_model.summary()
 
# Train on episodes
def train():
    for ep in range(epochs):
        epoch_loss    = 0
        epoch_acc     = 0
        epoch_loss_su = 0
        
        ## Get an episode for training model
        for idx in range(tr_iterations):
            # Train the main model
            support, smask, query, qmask = U.get_episode_FSS(setX= Train_list_FSS, n_way = NWAY, k_shot = KSHOT, data_path= FSS_add)
            acc_loss = model.train_on_batch([support, smask, query], qmask)
            # Train the Surrogate_model
            ISIC_unlabel            = U.get_episode_surrogate(IMG=Surrogate_set_ISIC, img_h = 224, img_w = 224, n_way = 1)
            FSS_unlabel             = support[:,0]
            noisy_data, target_data = U.add_noise(FSS_unlabel, ISIC_unlabel)
            Surrogate_loss          = Surrogate_model.train_on_batch([noisy_data], target_data)
            # Losses
            epoch_loss    += acc_loss[0]
            epoch_acc     += acc_loss[1]                
            epoch_loss_su += Surrogate_loss
            
            if (idx % 50) == 0:
                print ('Base_Model:::Epoch > ',(ep+1),' --- Iteration > ', (idx+1),'/',tr_iterations,' --- BM_Loss:', epoch_loss/(idx+1), ' --- Acc: ', epoch_acc/(idx+1))
                print ('Surrogate_Model:::Epoch > ',(ep+1),' --- Iteration > ', (idx+1),'/',tr_iterations,' --- AM_Loss:', epoch_loss_su/(idx+1))
            
        evaluate(ep)

def evaluate(ep):
    global Best_performance
    global Valid_Dice
    overall_Dice       = 0.0
   
    for idx in range (it_eval):
        ## Get an episode for evaluation 
        support, smask, query, qmask = U.get_episode_test(ph2_img, ph2_msk, img_h = 224, img_w = 224, n_way = NWAY, k_shot = KSHOT)
        Es_mask = model.predict([support, smask, query])
        
        Dice_score   = U.compute_dice(Es_mask, qmask)
        overall_Dice += Dice_score
        
    print('Epoch>>>', ep+1 ,'Dice score on Ph2 set>> ', overall_Dice/ it_eval)   
    Valid_Dice.append(overall_Dice / it_eval) 
    if Best_performance<(overall_Dice / it_eval):
       Best_performance = (overall_Dice / it_eval)
       model.save_weights('FSMS_model_weights.h5')
       
## Train and test the model
train()

Performance = {}
Performance['Valid_Dice']    = Valid_Dice

with open('FSMS_model_performance.pkl', 'wb') as f:
        pickle.dump(Performance, f, pickle.HIGHEST_PROTOCOL)
        
        