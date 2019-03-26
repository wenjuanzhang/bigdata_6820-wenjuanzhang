from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from collections import defaultdict
from functools import partial
import numpy as np
import pickle 
import argparse
arg_parser = argparse.ArgumentParser() 
arg_parser.add_argument('--layer',
                            dest='layer',
                            action="store",
                            type=int,
                            default=1,
                            help='layer')
args = arg_parser.parse_args()
print("args: ",args)

#
#@ Get the VGG16 model that comes with Keras
model = VGG16(weights='imagenet')
print(model.summary())
#
# Qucik test on an elephant poicture
img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_data = image.img_to_array(img)
img_data = np.expand_dims(img_data, axis=0)
img_data = preprocess_input(img_data)
#
# Actually run the model on the elep[hant picture
preds = model.predict(img_data)

# decode the results into a list of tuples (class, description, probability)
print('Predicted:', decode_predictions(preds, top=3)[0])
#
# Should get this result: Predicted: [('n02504013', 'Indian_elephant', 0.537905), ('n02504458', 'African_elephant', 0.28958666), ('n01871265', 'tusker', 0.14621235)]

#
# Get the list of validation files for the ILSVRC2012 challenge
import glob
files = glob.glob('/fs/scratch/PAS1495/physics6820/ILSVRC2012/*.JPEG')
#
# Here we can make a new model which has the same inoputs as the VGG model, but has
# an output at an intermediate layer
from keras.models import Model
#
# Thios info came from inspecting the output of print(model.summary()) above
#
# The output from the first convolutional layer
layer_name = 'block1_pool'
numxy = 112
filters = 64

layer = args.layer
if layer==3:
#
# The output from the middle convolutional layer
    layer_name = 'block3_pool'
    numxy = 28
    filters = 256
#
# The output from the last convolutional layer
elif layer==5:
    layer_name = 'block5_pool'
    numxy = 7
    filters = 512
#
# This defines a new model which has as its output the output of the above chosen layer
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)

print("Predicting outputs for layer ",layer_name)
#
# We use partial instead of autovivify because otherwise we can't pickle the output (not sure why)
neuronsByImageIndex = defaultdict(partial(defaultdict, float))
#
# Put each image in
numFiles = 0
for img_path in files:
    numFiles += 1
    if numFiles%100 == 0:
        print("Processed ",numFiles)
    if numFiles%3000 == 0:
        pickle.dump(neuronsByImageIndex,open('neuronsByImageIndex_'+layer_name+'.pkl', 'wb') )

    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    decoded_imgs = intermediate_layer_model.predict(img_data)
    count = 0
    countPix = 0
    for i3 in range(filters):
        neuronsByImageIndex[count][img_path] = decoded_imgs[0,:,:,i3].sum()
        count += 1
        for i1 in range(numxy):
            for i2 in range(numxy):
                if countPix < 100:
                    neuronsByImagePixelIndex[countPix][img_path] = decoded_imgs[0,i1,i2,i3].sum()
                    countPix += 1
#
# Save our dictionary
pickle.dump(neuronsByImageIndex,open('neuronsByImageIndex_'+layer_name+'.pkl', 'wb') )
print("Done!",count)
