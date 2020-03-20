# Abdur. R. Fayjie, R. Azad, Claude Kauffman, Ismail Ben Ayed, Marco Pedersoli and Jose Dolz "Semi-supervised Few-Shot Learning for Medical Image Segmentation", arXiv preprint arXiv, 2020
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import keras
import keras.layers as layers 
from keras.models import Model
from keras.layers.core import Lambda
import encoder_models as EM
import numpy as np
from keras import backend as K
    
def GlobalAveragePooling2D_r(f):
    def func(x):
        repc =  int(x.shape[4])
        m    =  keras.backend.repeat_elements(f, repc, axis = 4)
        x    =  layers.multiply([x, m])
        repx =  int(x.shape[2])
        repy =  int(x.shape[3])
        x    = (keras.backend.sum(x, axis=[1, 2, 3], keepdims=True) / (keras.backend.sum(m, axis=[1, 2, 3], keepdims=True)))
        x    =  keras.layers.Reshape(target_shape=(np.int32(x.shape[2]), np.int32(x.shape[3]), np.int32(x.shape[4])))(x)
        x    =  keras.backend.repeat_elements(x, repx, axis = 1)
        x    =  keras.backend.repeat_elements(x, repy, axis = 2)       
        return x
    return Lambda(func)
    
def common_representation(x1, x2):    
    x = layers.concatenate([x1, x2], axis=3) 
    x = layers.Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = layers.BatchNormalization(axis=3)(x) 
    x = layers.Activation('relu')(x) 
    return x

def GlobalAveragePooling2D_r2(f):
    def func(x):
        repc =  int(x.shape[3])
        m    =  keras.backend.repeat_elements(f, repc, axis = 3)
        x    =  layers.multiply([x, m])     
        return x
    return Lambda(func)
    
       
def my_model(encoder = 'VGG', input_size = (256, 256, 1), k_shot =1, learning_rate = 1e-4, learning_rate2 = 1e-4, no_weight = False):
    # Get the encoder
    if encoder == 'VGG':
       encoder = EM.vgg_encoder(input_size = input_size, no_weight = no_weight)
    else:
       print('Encoder is not defined yet')
       
    S_input  = layers.Input(input_size)
    Q_input  = layers.Input(input_size)
    ## Encode support and query sample
    s_encoded = encoder(S_input)

    ## Auxiliary task
    x1  = layers.Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(s_encoded)
    x1  = layers.BatchNormalization(axis=3)(x1)
    x1  = layers.Activation('relu')(x1) 
    x1  = layers.Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(x1)
    x1  = layers.Conv2D(3,  3, padding = 'same', kernel_initializer = 'he_normal')(x1)        
    xa  = layers.Activation('sigmoid')(x1)
    
    ###################################### K-shot learning #####################################
    ## K shot
    S_input2  = layers.Input((k_shot, input_size[0], input_size[1], input_size[2]))
    Q_input2  = layers.Input(input_size)
    S_mask2   = layers.Input((k_shot, int(input_size[0]/4), int(input_size[1]/4), 1))  
      
    kshot_encoder = keras.models.Sequential()
    kshot_encoder.add(layers.TimeDistributed(encoder, input_shape=(k_shot, input_size[0], input_size[1], input_size[2])))

    s_encoded = kshot_encoder(S_input2)
    q_encoded = encoder(Q_input2)
    s_encoded = layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))(s_encoded)
    q_encoded = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(q_encoded) 

    ## Global Representation
    s_encoded  = GlobalAveragePooling2D_r(S_mask2)(s_encoded)   

    ## Common Representation of Support and Query sample
    Bi_rep  = common_representation(s_encoded, q_encoded)

    ## Decode to query segment
    x = layers.Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(Bi_rep)
    x = layers.BatchNormalization(axis=3)(x) 
    x = layers.Activation('relu')(x)       
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = layers.BatchNormalization(axis=3)(x) 
    x = layers.Activation('relu')(x)    
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = layers.BatchNormalization(axis=3)(x) 
    x = layers.Activation('relu')(x)       
    x = layers.Conv2D(64, 3,  activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = layers.Conv2D(2, 3,   activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    final = layers.Conv2D(1, 1,   activation = 'sigmoid')(x)  
    
    seg_model = Model(inputs=[S_input2, S_mask2, Q_input2], outputs = final)

    seg_model.compile(optimizer = keras.optimizers.Adam(lr = learning_rate), loss = 'binary_crossentropy', metrics = ['accuracy']) 

    Surrogate_model = Model(inputs=[S_input], outputs = xa)
    Surrogate_model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(lr = learning_rate2))
              
    return seg_model, Surrogate_model    
    
    
    
    