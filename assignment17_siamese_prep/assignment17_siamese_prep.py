
import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model

from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers import Lambda, Flatten, Dense, Input
from keras.initializers import glorot_uniform,RandomNormal

from keras.engine.topology import Layer
from keras.regularizers import l2
from keras import backend as K
from keras import layers
import numpy as np

import time

import random
import numpy as np
from sklearn.utils import shuffle
import random
import numpy as np

import pickle

#
# Methods for defining and initializaing our model
def initialize_weights(shape, name=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer weights with mean as 0.0 and standard deviation of 0.01
    """
    return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)
def initialize_bias(shape, name=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
    """
    return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)


def get_siamese_model_works(input_shape):
    """
        Model architecture based on the one provided in: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
    """
    
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    
    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(64, (10,10), activation='relu', input_shape=input_shape,
                   kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7,7), activation='relu',
                     kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4,4), activation='relu', kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4,4), activation='relu', kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid',
                   kernel_regularizer=l2(1e-3),
                   kernel_initializer=initialize_weights,bias_initializer=initialize_bias))
    
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    
    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid',bias_initializer=initialize_bias)(L1_distance)
    
    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    
    # return the model
    return siamese_net

def get_siamese_model(input_shape):
    """
        Model architecture based on the one provided in: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
    """
    
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    
    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(64, (10,10), activation='relu', input_shape=input_shape,
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                   kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7,7), activation='relu',
                     kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                     bias_initializer=RandomNormal(mean=0.5, stddev=0.01), 
                     kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4,4), activation='relu', 
                     kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                     bias_initializer=RandomNormal(mean=0.5, stddev=0.01), 
                     kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4,4), activation='relu', 
                     kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                     bias_initializer=RandomNormal(mean=0.5, stddev=0.01), 
                     kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid',
                   kernel_regularizer=l2(1e-3),
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                    bias_initializer=RandomNormal(mean=0.5, stddev=0.01)))
    
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    
    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid',bias_initializer=RandomNormal(mean=0.5, stddev=0.01))(L1_distance)
    
    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    
    # return the model
    return siamese_net
#
# This make an N-way test sample
#    if use_test_data is True, uses validation data, otherwise uses training data
def make_oneshot(N,use_test_data=True):
#
# For each batch shuffle
    if use_test_data:
        n_classes, n_examples, w, h = X_val.shape
    else:
        n_classes, n_examples, w, h = X_train.shape
    letters = list(range(n_classes))
    examples = list(range(n_examples))
#
# Each trial shuffle our letters and classes 
    random.shuffle(letters)
    random.shuffle(examples)
    #print(examples)
#
# Get our support indices
    support_letters_class_indices = letters[0:N]
    support_letters_example_indices = examples[0:N]
#
# Get the letter to test
    test_letter_class_index = letters[0]
    test_letter_example_index = examples[0]
#
# Now get another example (but different) of the test letter
    test_letter_class_index_other = letters[0]
    test_letter_example_index_other = examples[1]
#
# The first letter in our support sample is the correct class
    support_letters_class_indices[0] = test_letter_class_index_other
    support_letters_example_indices[0] = test_letter_example_index_other
    targets = np.zeros((N,))
    targets[0] = 1
#
# Now form our images
    if use_test_data:
        test_images = np.asarray([X_val[test_letter_class_index,test_letter_example_index,:,:]]*N)
        test_images = test_images.reshape(N, w, h,1)
        support_images = X_val[support_letters_class_indices,support_letters_example_indices,:,:]
        support_images = support_images.reshape(N, w, h,1)
    else:
        test_images = np.asarray([X_train[test_letter_class_index,test_letter_example_index,:,:]]*N)
        test_images = test_images.reshape(N, w, h,1)
        support_images = X_train[support_letters_class_indices,support_letters_example_indices,:,:]
        support_images = support_images.reshape(N, w, h,1)
        
#
# Form return
    pairs = [test_images, support_images]

    return pairs, targets


def get_batch(batch_size):
#
# For each batch shuffle
    n_classes, n_examples, w, h = X_train.shape
    letters = list(range(n_classes))
    examples = list(range(n_examples))
#
# Each trial shuffle our letters and classes 
    random.shuffle(letters)
    random.shuffle(examples)
    targets = np.zeros((batch_size,))
    test_images = np.zeros((batch_size,w, h))
    support_images = np.zeros((batch_size,w, h))
#
# Make sure the batch size is < half the classes
    if batch_size < n_classes//2:
        half_batch = batch_size//2
#
# Get the indices for the 1st half - which are pairs from same class
        test_letter_class_indices = letters[0:half_batch]
        test_letter_example_index = examples[0]
        support_letters_class_indices = letters[0:half_batch]
        support_letters_example_index = examples[1]
        
        test_images[0:half_batch,:,:] = X_train[test_letter_class_indices,test_letter_example_index,:,:]
        support_images[0:half_batch,:,:] = X_train[support_letters_class_indices,support_letters_example_index,:,:]
        targets[0:half_batch] = 1
#
# Get the indices for the 2nd half - which are pairs from different classes
        test_letter_class_indices = letters[half_batch:batch_size]
        test_letter_example_index = examples[0]
        support_letters_class_indices = letters[batch_size:batch_size+half_batch]
        support_letters_example_index = examples[1]
        
        test_images[half_batch:batch_size,:,:] = X_train[test_letter_class_indices,test_letter_example_index,:,:]
        support_images[half_batch:batch_size,:,:] = X_train[support_letters_class_indices,support_letters_example_index,:,:]
        targets[half_batch:batch_size] = 0
#
# Reshape
        test_images = test_images.reshape(batch_size, w, h,1)
        support_images = support_images.reshape(batch_size, w, h,1)

#
# Now shuffle coherently
    targets, test_images, support_images = shuffle(targets, test_images, support_images)
    pairs = [test_images, support_images]

    return pairs, targets

#
# Get our training and validation data sets
X_train,lang_train = pickle.load(open("data/train.pickle","rb"))
X_val,lang_val = pickle.load(open("data/val.pickle","rb"))
#
# These need to be normalized
X_train = X_train / 255.0
X_val = X_val / 255.0
#
# Some debug printout
print(X_train.shape)
print(X_val.shape)
print(X_val[0,0,:,:])
print("Training:    ",lang_train)
print("Validation:  ",lang_val)
img = X_val[0,1,:,:]
print(img.shape)
#
# Set up the model
model = get_siamese_model((105, 105, 1))
optimizer = Adam(lr = 0.00006)
model.compile(loss="binary_crossentropy",optimizer=optimizer)
model.summary()
#
# Hyper parameters
evaluate_every = 200 # interval for evaluating on one-shot tasks
batch_size = 32
n_iter = 20000 # No. of training iterations
N_way = 20 # how many classes for testing one-shot tasks
n_val = 200 # how many one-shot tasks to validate on
best = -1
#
# Now start training
print("Starting training process!")
print("-------------------------------------")
t_start = time.time()
for i in range(1, n_iter+1):
#
# Get a new batch to test on
    (inputs,targets) = get_batch(batch_size)
    loss = model.train_on_batch(inputs, targets)
#
# Every so many iterations, check the training and validation performance
    if i % evaluate_every == 0:
        print("i=",i)
        print("\n ------------- \n")
        print("Time for {0} iterations: {1} mins".format(i, (time.time()-t_start)/60.0))
        print("Train Loss: {0}".format(loss)) 
#
# Now get N-way test results for train and validation sets
        n_correct_train = 0
        n_correct_val = 0
        for testTrials in range(n_val):
#
# First check training perfromance
            inputs, targets = make_oneshot(N_way,use_test_data=False)
            probs = model.predict(inputs)
            if np.argmax(probs) == np.argmax(targets):
                n_correct_train += 1
#
# Next check validation performace
            inputs, targets = make_oneshot(N_way,use_test_data=True)
            probs = model.predict(inputs)
            if np.argmax(probs) == np.argmax(targets):
                n_correct_val += 1
        train_acc = (100.0 * n_correct_train / n_val)
        print("     training perf",train_acc)
        val_acc = (100.0 * n_correct_val / n_val)
        print("   validation perf",val_acc)
        if val_acc >= best:
            print("Current best: {0}, previous best: {1}".format(val_acc, best))
            best = val_acc
            model.save('models/siam_model.{}.h5'.format(i))
            model.save_weights('weights/siam_weights.{}.h5'.format(i))

print("Done!")