# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 12:03:38 2023

@author: diego
"""

import keras.backend as K
import numpy as np
import tensorflow as tf

# import libraries
import numpy as np
import os
import pandas as pd
from PIL import Image
from matplotlib import pyplot


from os.path import abspath
from os import sep

from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD
from keras.initializers import RandomNormal
from keras.models import Model
from keras import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import ReLU


from datetime import datetime
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
import pydot
import graphviz 
from keras.utils import plot_model
from tensorflow.image import ssim


#%% load and dataset transform function



def ssim_loss(y_true, y_pred):
    #y_pred = tf.squeeze(y_pred)
    #print(y_pred)
    # y_true = tf.transpose(y_true, perm=[1, 2, 0,1])
    # y_pred = tf.transpose(y_pred, perm=[1, 2, 0])
    # calculate the ssim between true and predicted values
    ssim_value = ssim(y_true, y_pred,max_val= 1,filter_size=1,
                      filter_sigma=1,
                      k1=0.01,
                      k2=0.03)
    # return the negative ssim as the loss (to minimize it)
    loss = 1.0 - ssim_value
    return loss


def custom_Stddes(y_true, y_pred):
    Epsilon = (y_true - y_pred)
    Epsilon2 = tf.square(Epsilon)
    MeanEpsilon2 =tf.keras.backend.mean(Epsilon2)
    MeanEpsilon = tf.keras.backend.mean(Epsilon)
    loss = tf.math.sqrt(MeanEpsilon2 - tf.square(MeanEpsilon))
    loss = loss*100
 
    return loss



def load_data(List_images,List_target,images_quantity,input_image,target):
    im_input = []
    im_target = []
    
    for i in range(images_quantity):
        print(i)
        im1 = Image.open(input_image+'/'+List_images[i])
        im1 = np.array(im1.convert('L'))
        im2 = Image.open(target+'/'+ List_target[i])
        im2 = np.array(im2.convert('L'))
        im_input.append(im1)
        im_target.append(im2)
    im_input = np.array(im_input)
    im_target = np.array(im_target)
    dataset=[im_input,im_target]
    return dataset


# define the discriminator model
def define_discriminator(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # source image input
    in_src_image = Input(shape=image_shape)
    # target image input
    in_target_image = Input(shape=image_shape)
    # concatenate images channel-wise
    merged = Concatenate()([in_src_image, in_target_image])
    # C64
    d = Conv2D(64, (4,4), strides=(2,2), padding='same',
               kernel_initializer=init)(merged)
    d = ReLU()(d)#LeakyReLU(alpha=0.2)(d) #LeakyReLU
    # C128
    d = Conv2D(128, (4,4), strides=(2,2), padding='same',
               kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = ReLU()(d)
    # C256
    d = Conv2D(256, (4,4), strides=(2,2), padding='same', 
               kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = ReLU()(d)
    # C512
    d = Conv2D(512, (4,4), strides=(2,2), padding='same', 
               kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = ReLU()(d)
    # second last output layer
    d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = ReLU()(d)
    # patch output
    d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_src_image, in_target_image], patch_out)
    # compile model
    opt = Adam(learning_rate=0.0000001)#Adam , beta_1=0.5 , RMSprop
    model.compile(loss=['binary_crossentropy'], optimizer=opt, 
                  loss_weights=None)#'binary_crossentropy'
    
    return model
    
# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add downsampling layer
    g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', 
               kernel_initializer=init)(layer_in)
    # conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g
    
# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same',
                        kernel_initializer=init)(layer_in)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
    # merge with skip connection
    g = Concatenate()([g, skip_in])
    # relu activation
    g = Activation('relu')(g)
    return g

# define the standalone generator model
def define_generator(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    # encoder model -> Secuencia de Convolución y activación 
    #-> Encoder ascendente
    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    e6 = define_encoder_block(e5, 512)
    #e7 = define_encoder_block(e6, 512)
    # bottleneck, no batch norm and relu
    b = Conv2D(512, (4,4), strides=(2,2), padding='same', 
               kernel_initializer=init)(e6)
    b = Activation('relu')(b)
    # decoder model -> Secuencia de Convolución y activación 
    #-> Decoder descendente
    d1 = decoder_block(b, e6, 512)
    d2 = decoder_block(d1, e5, 512)
    d3 = decoder_block(d2, e4, 512)
    d4 = decoder_block(d3, e3, 256, dropout=False)
    d5 = decoder_block(d4, e2, 128, dropout=False)
    d6 = decoder_block(d5, e1, 64, dropout=False)
    #d7 = decoder_block(d6, e1, 64, dropout=False)
    # output
    g = Conv2DTranspose(image_shape[2], (4,4), strides=(2,2), padding='same', 
                        kernel_initializer=init)(d6)
    out_image = Activation('relu')(g) #'tanh'
    # define model
    model = Model(in_image, out_image)
    return model
    
# define the combined generator and discriminator model, for
# updating the generator
def define_gan(g_model, d_model, image_shape):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # define the source image
    in_src = Input(shape=image_shape)
    # connect the source image to the generator input
    gen_out = g_model(in_src)
    # connect the source input and generator output to the discriminator input
    dis_out = d_model([in_src, gen_out])
    # src image as input, generated image and classification output
    model = Model(in_src, [dis_out, gen_out])
    # compile model
    opt = RMSprop(lr=0.0000001)#, beta_1=0.5
    model.compile(loss=ssim_loss, optimizer=opt, 
                  loss_weights=None) #loss=custom_Stddes, loss=['mean_squared_error', 'mae']
    return model
    
# load and prepare training images
def load_real_samples(filename):
    # load the compressed arrays
    data = load(filename)
    # unpack the arrays
    X1, X2 = data['arr_0'], data['arr_1']
    # scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]
    
# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
    # unpack dataset
    trainA, trainB = dataset
    # choose random instances
    ix = randint(0, trainA.shape[0], n_samples)
    # retrieve selected images
    X1, X2 = trainA[ix], trainB[ix]
    # generate ✬real✬ class labels (1)
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y 
    
# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance
    X = g_model.predict(samples)
    # create ✬fake✬ class labels (0)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, dataset, n_samples=3):
    # select a sample of input images
    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
    # generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
    # scale all pixels from [-1,1] to [0,1]
    # X_realA = (X_realA + 1) / 2.0
    # X_realB = (X_realB + 1) / 2.0
    # X_fakeB = (X_fakeB + 1) / 2.0
    # plot real source images
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(X_realA[i],cmap='gray')
    # plot generated target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(X_fakeB[i],cmap='gray')
    # plot real target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
        pyplot.axis('off')
        pyplot.imshow(X_realB[i],cmap='gray')
    # save plot to file
    filename1 = 'plot_%06d.png' % (step+1)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = 'model_%06d.h5' % (step+1)
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))

# train pix2pix models
def train(d_model, g_model, gan_model, dataset, n_epochs, n_batch=1):
    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
    # unpack dataset
    trainA, trainB = dataset
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs
    for i in range(n_steps):
        # select a batch of real samples
        [X_realA, X_realB], y_real = generate_real_samples(dataset, 
                                                           n_batch, n_patch)
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA,
                                                n_patch)
        
        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        # update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        # update the generator
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        # summarize performance
        print('>%d/%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1,n_steps, d_loss1, 
                                                     d_loss2, g_loss))
        d_loss1_val.append(d_loss1)
        d_loss2_val.append(d_loss2)
        g_loss_val.append(g_loss)
        n_steps_val.append(i)
        
        #summarize model performance
        if (i+1) % (bat_per_epo * 1) == 0:
            summarize_performance(i, g_model, dataset)


#%% Read data
path = 'C:/Users/diego/OneDrive - Universidad EAFIT/Eafit/Phase unwrapping/data/'
input_image = path+'wrapped_simple'
target = path+'real_simple'
List_images = os.listdir(input_image)
List_target = os.listdir(target)
sample = Image.open(input_image+'/'+List_images[0])
sample = np.array(sample.convert('L'))
image_shape = (np.shape(sample)[0], np.shape(sample)[1], 1)
images_quantity = 200#len(List_images)
dataset = load_data(List_images,List_target,images_quantity,input_image,target)
dataset[0]=dataset[0]/255
dataset[0] = dataset[0].astype('float32')
dataset[1]=dataset[1]/255
dataset[1] = dataset[1].astype('float32')

#%% define the models

d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)

#%% Plot model arquitecture
img_filed= path +'d_model.png'
img_fileg = path +'g_model.png'
img_filegan = path +'gan_model.png'

plot_model(d_model, to_file=img_filed, 
           show_shapes=True, show_layer_names=True)
plot_model(g_model, to_file=img_fileg, show_shapes=True, 
           show_layer_names=True)
plot_model(gan_model, to_file=img_filegan, 
           show_shapes=True, show_layer_names=True)
#%%
d_loss1_val = []
d_loss2_val = []
g_loss_val  = []
n_steps_val = []
n_epochs = 50

train(d_model, g_model, gan_model, dataset,n_epochs)
#%%
figure, axs = pyplot.subplots(1,2)
axs[0].plot(n_steps_val,d_loss1_val,label='Real Data')
axs[0].plot(n_steps_val,d_loss2_val,label='fake Data')
axs[1].plot(n_steps_val,g_loss_val,label='Generator')
axs[0].legend()
axs[1].legend()

#%%
[X_realA, X_realB], y_real = generate_real_samples(dataset, 
                                                   3, 1)
X_fakeB, y_fake = generate_fake_samples(g_model, X_realA,
                                       3)
#%%
X_fakeB = np.squeeze(X_fakeB)
#%%
# y_true=tf.convert_to_tensor(X_realA, dtype=None, dtype_hint=None, name=None)
# y_pred=tf.convert_to_tensor(X_fakeB, dtype=None, dtype_hint=None, name=None)

def Stddes(y_true, y_pred):
    Epsilon = (y_true - y_pred)
    Epsilon2 = Epsilon**2
    MeanEpsilon2 = np.mean(Epsilon2)
    MeanEpsilon = np.mean(Epsilon)
    loss = np.sqrt(MeanEpsilon2 - (MeanEpsilon)**2)*100
    return loss, Epsilon


def ssim_index(img1, img2):
    # Convert images to tensors
    # img1 = tf.convert_to_tensor(img1, dtype=tf.float32)
    # img2 = tf.convert_to_tensor(img2, dtype=tf.float32)
    # Compute SSIM index
    ssim = tf.image.ssim(img1, img2, max_val=1)
    # Return SSIM index as float value
    return ssim.numpy()

img = 2
y_true = X_realA
y_pred = X_realB
y_true = np.transpose(y_true, axes=[1, 2, 0])
y_pred = np.transpose(y_pred, axes=[1, 2, 0])
# y_true = np.expand_dims(X_realB[0],axis=2)
# y_pred = np.expand_dims(X_fakeB[0],axis=2)
#y_pred = X_realB[img,:,:]
#loss,Epsilon = Stddes(y_true, y_pred)
a = ssim_index(y_true, y_pred)
b = 1-ssim_index(y_true, y_pred)
#%%
# loss = ssim_index(y_true, y_pred)
fig,axs= pyplot.subplots(1,2)
axs[0].imshow(X_realA[img,:,:],cmap='gray')
axs[1].imshow(X_realB[img,:,:],cmap='gray')
# print('Desviation = ',loss)

#%%
pyplot.imshow(X_realA[0][:,:],cmap='gray')



#%%
# =============================================================================
# #X_fakeB = X_fakeB.astype('uint8')
# img = 0
# fig,axs= pyplot.subplots(1,2)
# axs[0].imshow(X_fakeB[img,:,:],cmap='gray')
# axs[1].imshow(X_realB[img,:,:],cmap='gray')
# =============================================================================
# =============================================================================
# y = np.ones((1024,1024))
# custo = wrapped_img*(-np.pi/2)
# pyplot.imshow(custo)
# loss = custom_Stddes(wrapped_img, custo)
# loss
# =============================================================================
