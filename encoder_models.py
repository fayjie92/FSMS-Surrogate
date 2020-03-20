# Abdur. R. Fayjie, R. Azad, Claude Kauffman, Ismail Ben Ayed, Marco Pedersoli and Jose Dolz "Semi-supervised Few-Shot Learning for Medical Image Segmentation", arXiv preprint arXiv, 2020
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import keras
import keras.layers as layers 
from keras.models import Model
import keras.backend as K
from keras.applications.vgg16 import VGG16
import numpy as np

############################################ Encoder Weights on Image Net ###########################################
VGG_WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                           'releases/download/v0.1/'
                           'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

################################## VGG 16 Encoder #######################################
def vgg_encoder(input_size = (256, 256, 3), no_weight = False):
    img_input = layers.Input(input_size) 
    modelvgg = VGG16(weights='imagenet', include_top=False)
    block1_conv1 = modelvgg.get_layer('block1_conv1').get_weights()
    weights, biases = block1_conv1
    weights = np.transpose(weights, (3, 2, 0, 1))
    kernel_out_channels, kernel_in_channels, kernel_rows, kernel_columns = weights.shape
    grayscale_weights = np.zeros((kernel_out_channels, 1, kernel_rows, kernel_columns))
    gray_w = (weights[:,0,:,:]* 0.2989) + (weights[:,1,:,:] * 0.5870) + (weights[:,2,:,:] * 0.1140)
    gray_w = np.expand_dims(gray_w, axis=1)
    grayscale_weights = gray_w
    grayscale_weights = np.transpose(grayscale_weights, (2, 3, 1, 0))
    
    # Block 1
    if input_size[2] == 1:
       xblock1 = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1_gray')                    
       x = xblock1(img_input) 
    else:
       x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
                                        
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same', dilation_rate=(2, 2),
                      name='block3_conv3')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)

    # Create model.
    model = Model(img_input, x, name='vgg16_model_with_block1-4')

    # Load weights.
    weights_path = keras.utils.get_file(
                   'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                   VGG_WEIGHTS_PATH_NO_TOP,
                   cache_subdir='models',
                   file_hash='6d6bbae143d832006294945121d1f1fc')
    if not(no_weight):
       model.load_weights(weights_path, by_name=True)
    if input_size[2] == 1:   
       xblock1.set_weights([grayscale_weights, biases])
    return model
