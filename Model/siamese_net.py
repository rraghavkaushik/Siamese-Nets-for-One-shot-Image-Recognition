import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense
from keras import Sequential, Model
from keras.initializers import RandomNormal
from keras.regularizers import l2
from keras.optimizers import SGD
import keras.backend as bknd
# from keras.activations import ReLU

np.random.seed(42)
random.seed(42)
    
    
def l1_distance(h1, h2):

    ''' 
    In the paper, they have used the weighted L1 distance, but 
    I couldn't find more information about
    the weights. So I used the standard L1 distance. 
    '''

    return bknd.sum(abs(h1 - h2))

class model():
    def __init__(self, initial_lr = 0.001, batch_size = 32):
        self.lr = initial_lr
        self.batch_size = batch_size()
        self.siamese_net()

    def siamese_net(self):
        W_init = RandomNormal(mean = 0, stddev = 0.01)
        b_init = RandomNormal(mean = 0.5, stddev = 0.01)
        W_init_fc = RandomNormal(mean = 0, stddev = 0.02)

        input_shape = (105, 105, 1)
        h1 = Input(input_shape)
        h2 = Input(input_shape)

        c = Sequential()
        c.add(Conv2D(64, (10, 10), strides = (1, 1), activation = 'relu', input_shape = input_shape, kernel_initializer = W_init, bias_initializer = b_init, kernel_regularizer = l2(2e-4)))
        c.add(MaxPool2D())
        c.add(Conv2D(128, (7, 7), strides = (1, 1), activation = 'relu', kernel_initializer = W_init, bias_initializer = b_init, kernel_regularizer = l2(2e-4)))
        c.adD(MaxPool2D())
        c.add(Conv2D(128, (4, 4), strides = (1, 1), activation = 'relu', kernel_initializer = W_init, bias_initializer = b_init, kernel_regularizer = l2(2e-4)))
        c.add(MaxPool2D())
        c.add(Conv2D(256, (4, 4), strides = (1, 1), activation = 'relu', kernel_initializer = W_init, bias_initializer = b_init, kernel_regularizer = l2(2e-4)))
        c.add(Flatten())
        c.add(Dense(4096, activation = 'sigmoid', kernel_initializer = W_init_fc, bias_initializer = b_init, kernel_regularizer = l2(1e-3)))
        
        h1_encoded = c(h1)
        h2_encoded = c(h2)
        distance = l1_distance(h1_encoded, h2_encoded)

        prediction = Dense(1, activation = 'sigmoid')(distance)

        self.model = Model(inputs = [h1, h2], outputs = prediction)
        opt = SGD(lr = 0.001, momentum = 0.5)
        #opt = Adam()

        self.model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ["accuracy"])







