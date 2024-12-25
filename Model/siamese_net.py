import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import pickle as pkl
import matplotlib.image as img
import time
import keras.backend as bknd

from skimage.transform import AffineTransform, warp, rotate
from sklearn.utils import shuffle

from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense
from keras import Sequential, Model
from keras.initializers import RandomNormal
from keras.regularizers import l2
from keras.optimizers import SGD

from statistics import mean
# from keras.activations import ReLU

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)


def apply_affine_transformation(image):

    affine_transform = AffineTransform(translation = (-30, 0))
    tranformed_image = warp(image, affine_transform, mode = 'warp')
    return tranformed_image

def rotate_clockwise(image):

    rotated_angle = random.randint(-45, 0)
    rotated_image = rotate(image, rotated_angle)
    return rotated_image

def rotate_anticlockwise(image):

    rotation_angle = random.randint(0, 45)
    rotated_image = rotate(image, rotation_angle)
    return rotated_image

def transform(image):

    # transformations = [apply_affine_transformation, rotate_clockwise, rotate_anticlockwise]
    # weights = [0.5, 0.5, 0.5]

    # selected_transformations = random.choices(transformations, weights = weights, k = 3)

    # for func in selected_transformations:
    #     image = func(image)

    # return image

    transformations = {
        "affine": apply_affine_transformation,
        "clockwise": rotate_clockwise,
        "anticlockwise": rotate_anticlockwise
    }

    for name, func in transformations.items():

        if random.random() > 0.5:
            image = func(image)

    return image

def pkl_data(filename):

    with open(filename, 'rb') as f:
        X_t, y_t = pkl.load(f)
    
    return X_t, y_t

def l1_distance(h1, h2):

    ''' 
    In the paper, they have used the weighted L1 distance, but 
    I couldn't find more information about
    the weights. So I used the standard L1 distance. 
    '''

    return bknd.sum(abs(h1 - h2))


class DataGenerator:

    def __init__(self, batch_size = 32, augment = True):
        self.batch_size = batch_size
        self.augment = augment

    def load_data(self, training_file):

        with open(training_file, 'rb') as f:
            return pkl.load(f)
        
    def apply_transform(self, image):

        if random.random() > 0.5:
            return transform(image)
        
        return image
        
    def process_batch(self, X_left, X_right):

        '''
            Applies transformation and preprocessing to batch of image.

            Arguments:
            - X_left: Array of left image file paths.
            - X_right: Array of right image file paths.

            Returns:
            - Preprocessed left and right image batches as NumPy arrays.
        '''

        X_left_batch = []
        X_right_batch = []

        for left_path, right_path in zip(X_left, X_right):

            left_image = img.imread(left_path)
            right_image = img.imread(right_path)

            if self.augment:
                left_image = self.apply_transform(left_image)
                right_image = self.apply_transform(right_image)

            X_left_batch.append(np.expand_dims(left_image, axis = 2))
            X_right_batch.append(np.expand_dims(right_image, axis = 2))

        return np.asarray(X_left_batch), np.asarray(X_right_batch)

        
    def load_batch(self, training_file, chunk_size = 1024):

        ''''
            This is a generator function to yield mini-batches of preprocessed data
            for training.
        '''

        X, y = self.load_data(training_file)
        data_len = len(X)

        while True:

            for chunk_start in range(0, data_len, chunk_size):
                
                chunk_end = min(chunk_start + chunk_end, data_len)
                X_chunk, y_chunk = X[chunk_start : chunk_end], y[chunk_start : chunk_end]
                X_chunk, y_chunk = shuffle(X_chunk, y_chunk, len(X_chunk))

                for batch_start in range(0, len(X_chunk), self.batch_size):

                    batch_end = min(batch_start + self.batch_size, len(X_chunk))

                    X_left, X_right, y_batch = (
                        X_chunk[batch_start : batch_end, 0],
                        X_chunk[batch_start : batch_end, 1],
                        y_chunk[batch_start : batch_end]
                    )

                    X_left_batch, X_right_batch = self.process_batch(X_left, X_right)

                    yield [X_left_batch, X_right_batch], y_batch
        

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


    def test_one_shot(self, X_left, X_right, y):

        prob = self.model.predict(X_left, X_right)
        return 1 if np.argmax(prob) == np.argmax(y) else 0


    def test_pairs(self, file_name, n_way = 20):

        correct_pred = 0
        X, y = pkl_data(file_name)

        j = 0
        for i in range(0, len(X), n_way):

            X_left, X_right, y = X[i : i + n_way, 0], X[i : i + n_way, 1], y[i : i + n_way]

            X_left, X_right, y = np.array(X_left), np.array(X_right), np.array(y)

            correct_pred += self.test_one_shot(X_left, X_right, y)

        n = len(X) / n_way
        accuracy = correct_pred * 100 / n

        return accuracy
    
    def test_validation_accuracy(self, wA_file, uA_file, n_way = 20):

        accuracy_wA = self.test_pairs(wA_file, n_way)
        accuracy_uA = self.test_pairs(uA_file, n_way)
        return accuracy_wA, accuracy_uA

    def train(self, load_model = False, best_accuracy = 0):

        wA_file = ''
        uA_file = ''

        with open("model.json", "w") as f:
            f.write(self.model.to_json())

        self.best_acc = best_accuracy
        self.val_acc = []
        self.train_metrics = []
        self.bknd = 0
        self.start = 1
        self.model_details = {'acc': 0, 'iter': 0, 
                              'model_lr': 0.0, 'model_mm': 0.0}
        self.val_acc_filename = 'val_acc'


        # if load_model == "True":

        data_generator = DataGenerator(self.batch_size, augment = True)
        train_generator = DataGenerator.load_batch()

        for i in range(self.start, 1000000):
            start_time = time.time()
            X_batch, y_batch = next(train_generator)
            loss = self.train(X_batch, y_batch)
            train_loss, train_acc = loss[0], loss[1]

            self.train_metrics.append([train_loss, train_acc])

            if i % 500:

                avg_train_loss = mean([metric[0] for metric in self.train_metrics])
                avg_train_acc = mean([metric[1] for metric in self.train_metrics])

                val_acc = self.test_validation_accuracy(wA_file, uA_file, n_way = 20)
                self.v_acc.append(val_acc)

                if val_acc[0] > self.best_accuracy:
                    print("\n***Saving model***\n")
                    self.model.save_weights("best_model/best_model.h5")
                    self.model_details.update({
                        'acc': val_acc[0],
                        'iter': i,
                        'model_lr': bknd.get_value(self.model.optimizer.learning_rate),
                        'model_mm': bknd.get_value(self.model.optimizer.momentum)
                    })
                    self.best_acc = val_acc[0]

                    with open(self.val_acc_filename, 'wb') as f:
                        pkl.dump((self.v_acc, self.train_metrics), f)
                    with open('best_model/model_details.pkl', 'wb') as f:
                        pkl.dump(self.model_details, f)

                end_time = time.time()
                print(f"Iteration: {i}  lr : {bknd.get_value(self.model.optimizer.learning_rate):.8f} "
                      f"momentum: {bknd.get_value(self.model.optimizer.momentum):.6f} avg_loss: {avg_train_loss:.4f} "
                      f"avg_acc: {avg_train_acc:.4f} wA_acc: {val_acc[0]:.2f}% uA_acc: {val_acc[1]:.2f}% "
                      f"time_taken: {end_time - start_time:.2f}s"
                )

                self.train_metrics = []
    


if __name__ == "__main__":

    model = model(batch_size = 32)






