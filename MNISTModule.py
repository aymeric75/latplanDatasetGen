#!/usr/bin/env python3

import numpy as np
#from .model.puzzle import *
#from .util import preprocess
import os
from skimage.transform import resize

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


#
#   .model.puzzle import *
#


#
# from .util import preprocess
#

def normalize(image):
    # into 0-1 range
    if image.max() == image.min():
        return image - image.min()
    else:
        return (image - image.min())/(image.max() - image.min())

def equalize(image):
    from skimage import exposure
    return exposure.equalize_hist(image)

def enhance(image):
    return np.clip((image-0.5)*3,-0.5,0.5)+0.5

def preprocess(image):
    image = image.astype(float)
    image = equalize(image)
    image = normalize(image)
    image = enhance(image)
    return image






setting = {
    'base' : None,
    'panels' : None,
    'loader' : None,
    'min_threshold' : 0.0,
    'max_threshold' : 0.5,
}

def load(width,height,force=False):
    if setting['panels'] is None or force is True:
        setting['panels'] = setting['loader'](width,height)


def generate(configs, width, height, **kwargs):

    load(width, height)

    def build():
        base = setting['base']
        P = len(setting['panels'])

        # Define the custom model class with the input shape
        class CustomModel(nn.Module):
            def __init__(self):
                super(CustomModel, self).__init__()
                self.configs_input = nn.Linear(P, P)  # Adjust input shape as needed
                self.base = base
                self.width = width
                self.height = height

            def forward(self, x):

                # first item is [7 5 6 .... 1 2 3]

                # One-hot encode configs
                configs_one_hot = F.one_hot(x.to(torch.int64), num_classes=self.width*self.height)
                matches = configs_one_hot.permute(0, 2, 1)
                print("matcheees")
                #matches = configs_one_hot.permute(0, 1, 2)
                
                
                matches = matches.reshape(-1, P)
              
                # Define panels as a PyTorch tensor
                panels = torch.tensor(setting['panels'], dtype=torch.float32)
                panels = panels.reshape(P, self.base * self.base)

                # Cast matches to the same data type as panels
                matches = matches.to(panels.dtype)

                # Calculate states
                states = torch.matmul(matches, panels)
                states = states.reshape(-1, self.height, self.width, self.base, self.base)
                states = states.permute(0, 1, 3, 2, 4)
                states = states.reshape(-1, self.height * self.base, self.width * self.base)
                #return wrap(x, states)
                return states

        model = CustomModel()

        return model

    model = build()
    
    # Convert configs to a NumPy array if it's not already
    if not isinstance(configs, np.ndarray):
        configs = np.array(configs)
    
    # Convert NumPy array to a PyTorch tensor
    configs_tensor = torch.tensor(configs, dtype=torch.float32)

    # Predict with the model
    with torch.no_grad():
        predictions = model(configs_tensor, **kwargs)

    return predictions.numpy()



def states(width, height, configs=None, **kwargs):
    digit = width * height
   
    if configs is None:
        configs = generate_configs(digit)
    return generate(configs,width,height, **kwargs)



def mnist (labels = range(10)):
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = (x_train.astype('float32') / 255.).round()
    x_test = (x_test.astype('float32') / 255.).round()
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    def conc (x,y):
        return np.concatenate((y.reshape([len(y),1]),x),axis=1)
    def select (x,y):
        selected = np.array([elem for elem in conc(x, y) if elem[0] in labels])
        return np.delete(selected,0,1), np.delete(selected,np.s_[1::],1).flatten()
    x_train, y_train = select(x_train, y_train)
  
    x_test, y_test = select(x_test, y_test)
    # plot_image(x_train[0], "im0.png")
    # plot_image(x_train[1], "im1.png")
  
    return x_train, y_train, x_test, y_test



def setup():
    setting['base'] = 16

    def loader(width,height):
        
        base = setting['base']
        x_train, y_train, _, _ = mnist()
       

        filters = [ np.equal(i, y_train) for i in range(9) ]
        print(np.array(filters).shape) # (9, 60000) # for each number (0 to 9), all the truth values over the dataset of 6k image

        imgs    = [ x_train[f] for f in filters ]


        # parcours de chaque numéro, et on prend la 1ière image imgs[0] qu'on reshape de 784 à 28x28
        panels  = [ imgs[0].reshape((28,28)) for imgs in imgs ]


        panels[8] = imgs[8][3].reshape((28,28))
        panels[1] = imgs[1][3].reshape((28,28))

        panels = np.array([resize(panel, (setting['base'], setting['base'])) for panel in panels])

      
        panels = preprocess(panels)


        return panels

    setting['loader'] = loader
