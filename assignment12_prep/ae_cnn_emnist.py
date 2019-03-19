from keras.datasets import mnist
import numpy as np
import scipy.io as sio
#
# See this for more info: https://arxiv.org/pdf/1702.05373.pdf
mat = sio.loadmat('/fs/scratch/PAS1043/physics6820/emnist/matlab/emnist-byclass.mat')
#print(mat)

data = mat['dataset']

ex_train = data['train'][0,0]['images'][0,0]
ey_train = data['train'][0,0]['labels'][0,0]
ex_test = data['test'][0,0]['images'][0,0]
ey_test = data['test'][0,0]['labels'][0,0]

ex_train = ex_train.reshape( (ex_train.shape[0], 28,28), order='F')
ex_test = ex_test.reshape( (ex_test.shape[0], 28,28), order='F')

ex_train = ex_train.reshape( (ex_train.shape[0], 784))
ex_test = ex_test.reshape( (ex_test.shape[0], 784))
ex_train = ex_train.astype('float32') / 255.
ex_test = ex_test.astype('float32') / 255.

import numpy as np
import scipy.io as sio
#
# Unpack the EMNIST data
#
# See this for more info: https://arxiv.org/pdf/1702.05373.pdf
mat = sio.loadmat('/fs/scratch/PAS1043/physics6820/emnist/matlab/emnist-byclass.mat')
#print(mat)

data = mat['dataset']

ex_train = data['train'][0,0]['images'][0,0]
ey_train = data['train'][0,0]['labels'][0,0]
ex_test = data['test'][0,0]['images'][0,0]
ey_test = data['test'][0,0]['labels'][0,0]

ex_train = ex_train.reshape( (ex_train.shape[0], 28,28), order='F')
ex_test = ex_test.reshape( (ex_test.shape[0], 28,28), order='F')

ex_train = ex_train.reshape( (ex_train.shape[0], 784))
ex_test = ex_test.reshape( (ex_test.shape[0], 784))
ex_train = ex_train.astype('float32') / 255.
ex_test = ex_test.astype('float32') / 255.

#
# Pull out the digits only for training and testing
import pandas as pd

df_train = pd.DataFrame(ex_train)
df_train['label'] = ey_train
df_digits_train = df_train[df_train['label']<=9]
x_train = df_digits_train.iloc[:,:784].values
x_train = x_train.reshape((x_train.shape[0],28,28,1))
y_train = df_digits_train['label'].values

df_test = pd.DataFrame(ex_test)
df_test['label'] = ey_test
df_digits_test = df_test[df_test['label']<=9]
x_test = df_digits_test.iloc[:,:784].values
x_test = x_test.reshape((x_test.shape[0],28,28,1))
y_test = df_digits_test['label'].values

#
# One host encode
from keras.utils import to_categorical
y_train_labels_cat = to_categorical(y_train)
y_test_labels_cat = to_categorical(y_test)

#
# How many of each category do we have?
unique, counts = np.unique(y_train, return_counts=True)
for digit,count in zip(unique, counts):
    print("digit",digit,"; count ",count)

from keras import models
from keras import layers
from keras import regularizers

# make our encoder
encoder = models.Sequential()
#
# First convolutional layer
encoder.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same',input_shape=(28,28,1)))
encoder.add(layers.MaxPooling2D((2,2), padding='same'))
encoder.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
encoder.add(layers.MaxPooling2D((2,2), padding='same'))
encoder.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
encoder.add(layers.MaxPooling2D((2,2), padding='same'))
print("encoder===>")
print(encoder.summary())

#
# Now make the decoder
# make our encoder
decoder = models.Sequential()
#
# First convolutional layer
decoder.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same',input_shape=(4,4,8)))
decoder.add(layers.UpSampling2D((2,2)))
decoder.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
decoder.add(layers.UpSampling2D((2,2)))
decoder.add(layers.Conv2D(16, (3, 3), activation='relu'))
decoder.add(layers.UpSampling2D((2,2)))
decoder.add(layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
print("decoder===>")
print(decoder.summary())

#
# Combine the encoder and decoder
autoencoder = models.Sequential()
autoencoder.add(encoder)
autoencoder.add(decoder)
#
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy',metrics=['mse'])
#autoencoder.compile(optimizer='adadelta', loss='mse',metrics=['mse'])
print("autoencoder===>")
print(autoencoder.summary())

#
# Actually run the fitter
history = autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

#
# Save our models
encoder.save('fully_trained_encoder_cnn.h5')
decoder.save('fully_trained_decoder_cnn.h5')
autoencoder.save('fully_trained_autoencoder_cnn.h5')

#
# Save our history data
import pickle 
pickle.dump(history.history,open('history_cnn.pkl', 'wb') )

