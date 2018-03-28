import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from os import getcwd
import csv
import tensorflow as tf
import argparse as _argparse
import pandas as pd
import seaborn as sb
import os
import h5py
import random
import glob


from keras.models import Sequential, model_from_json, load_model
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.layers import Cropping2D
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def random_brightness(image):
    """
    Returns an image with a random degree of brightness.
    :param image: Image represented as a numpy array.
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    brightness = .25 + np.random.uniform()
    image[:,:,2] = image[:,:,2] * brightness
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image

def preprocess_image(image):
    """
    Returns an image after applying several preprocessing functions.
    :param image: Image represented as a numpy array.
    """
    image = image[40:-20,:] #crop image
    image = random_brightness(image) #add random brightness adjustment
    image = cv2.resize(image, (200, 66), interpolation=cv2.INTER_AREA) #resize image
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    return image


#build the generator
def generator(samples, batch_size=32):
    """
    samples: a pandas dataframe containing training sample
    batch_size: the batch size
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples.iloc[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples.iterrows(): #converte pd dataframe to iterable
                #print(batch_sample)
                center_name = batch_sample[1][0].split('/')[-1]
                path = 'drive_data0/IMG/' + center_name
                
                center_image = cv2.imread(path)
                center_angle = float(batch_sample[1][3])
                
                #add random brightness
                center_image = random_brightness(center_image)
                
                # Flip image and apply opposite angle with probability of .5
                if random.randrange(2) == 1:
                    center_image = cv2.flip(center_image, 1)
                    center_angle = -center_angle
                
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)
            

#nvidia model
def create_model_nvidia():
    """Define the CNN architecture"""
    model = Sequential()  
    
    # Normalize  
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(66, 200, 3)))
    
    #crop
    #model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    #model.add(Lambda(resize_images, input_shape=(66, 200, 3)))
    
    # Add three 5x5 convolution layers (output depth 24, 36, and 48), each with 2x2 stride  
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))  
    model.add(ELU())  
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))  
    model.add(ELU())  
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))  
    model.add(ELU())  
    #model.add(Dropout(dropout))
    
    # Add two 3x3 convolution layers (output depth 64, and 64)  
    model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))  
    model.add(ELU())  
    model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))  
    model.add(ELU())  
    
    # Add a flatten layer  
    model.add(Flatten())  
    
    # Add three fully connected layers (depth 100, 50, 10 neurons respectively), elu activation
    model.add(Dense(100, W_regularizer=l2(0.001)))  
    model.add(ELU())  
    
    #model.add(Dropout(dropout))
    model.add(Dense(50, W_regularizer=l2(0.001)))  
    model.add(ELU())  
    
    #model.add(Dropout(dropout))
    model.add(Dense(10, W_regularizer=l2(0.001)))  
    model.add(ELU())  
    
    #model.add(Dropout(dropout))
    # Add a fully connected output layer  
    model.add(Dense(1))  
    # Compile and train the model,   
    model.compile(optimizer="adam", loss="mse")
    
    return model

if __name__=="__main__":
    tf.app.flags._global_parser = _argparse.ArgumentParser() #reset flags
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    
    flags.DEFINE_string('data_path', 'drive_data0/driving_log.csv', 'The path to the csv of training data.')
    flags.DEFINE_string('save_dir', 'models/', 'The directory to which to save the model.')
    flags.DEFINE_integer('batch_size', 128, 'The minibatch size.')
    flags.DEFINE_integer('num_epochs', 10, 'The number of epochs to train for.')
    flags.DEFINE_float('lrate', 0.00001, 'The learning rate for training.')
    
    driving_log = combine_data() #read in all the csvs
    driving_log = randomly_drop_low_steering_data(driving_log)

    # Split train and validation data
    train, test = train_test_split(driving_log, test_size=0.05, random_state=42)
    
    # compile and train the model using the generator function
    train_generator = generator(train, batch_size=32)
    validation_generator = generator(test, batch_size=32)

    #build the model
    model = create_model_nvidia()
    # model.compile(loss='mse', optimizer='adam')
    history_object = model.fit_generator(train_generator, #save the model object so you can run analysis on it later
                        samples_per_epoch= len(train), 
                        validation_data=validation_generator, 
                        nb_val_samples=len(test), 
                        nb_epoch=1)

    #save the model
    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)

    json = model.to_json()
    #model.save_weights(os.path.join(FLAGS.save_dir, 'model.h5'))
    model.save(os.path.join(FLAGS.save_dir, 'model.h5'))
    with open(os.path.join(FLAGS.save_dir, 'model.json'), 'w') as f:
        f.write(json)
    
