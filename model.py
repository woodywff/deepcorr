import pdb
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (Layer, Dense, Conv2D, MaxPool2D, Dropout, Flatten,
                                     GlobalAveragePooling2D, ZeroPadding2D)
import tensorflow as tf
import numpy as np

import yaml
with open('config.yml') as f:
    FLAG_DEBUG = yaml.load(f,Loader=yaml.FullLoader)['FLAG_DEBUG']

class DeepCorrCNN(Model):
    def __init__(self, conv_filters, dense_layers, drop_p):
        '''
        conv_filters: filters for the first two conv layers
        dense_layers: units for the last dense layers
        drop_p: dropout rate
        '''
        super().__init__(self)
        self.convs = Sequential([Conv2D(conv_filters[0], [2,20], strides=2, activation='relu'),
                                MaxPool2D([1,5]),
                                Conv2D(conv_filters[1], [4,10], strides=2, activation='relu'),
                                MaxPool2D([1,3])])
        self.flatten = Flatten()
        self.dense = Sequential()
        for i,units in enumerate(dense_layers):
            self.dense.add(Dense(units, activation=('relu' if i < len(dense_layers)-1 else None)))
            if i < len(dense_layers)-2:
                self.dense.add(Dropout(drop_p))
    def call(self, x):
        x = self.convs(x)
        x = self.flatten(x)
        y = self.dense(x)
        if FLAG_DEBUG:
            self.convs.summary()
            self.dense.summary()
        return y
        
        
      
        
        
       

    