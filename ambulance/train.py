import os
import numpy
import argparse
import random
import cv2
import matplotlib as mat
from sklearn.model_selection import train_test_split
import imutils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from imutils import paths
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical

import matplotlib.pyplot as matplot
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

mat.use("Agg")


def build(width, height, depth, classes):
    # initializing the model
    modelRequired = Sequential()
    shapeInput = (height, width, depth)

    # if we are using "channels first", update the input shape
    if K.image_data_format() == "channels_first":
        shapeInput = (depth, height, width)

    modelRequired.add(Conv2D(60, (5, 5), input_shape=shapeInput, activation='relu'))
    modelRequired.add(Conv2D(60, (5, 5), activation='relu'))
    modelRequired.add(MaxPooling2D(pool_size=(2, 2)))

    modelRequired.add(Conv2D(30, (3, 3), activation='relu'))
    modelRequired.add(Conv2D(30, (3, 3), activation='relu'))
    modelRequired.add(MaxPooling2D(pool_size=(2, 2)))

    modelRequired.add(Flatten())
    modelRequired.add(Dense(500, activation='relu'))
    modelRequired.add(Dropout(0.2))
    modelRequired.add(Dense(classes, activation='softmax'))

    # return the network architecture constructed
    return modelRequired


# Creating an argument parser to parse in the arguments
ap = argparse.ArgumentParser()

# Takes in the address of the dataset the model is to be trained upon
ap.add_argument("-i", "--images", required=True, help="path to input dataset")

# Takes in the address where the model will be installed
ap.add_argument("-m", "--modelToBeSaved", required=True, help="path to output model")
args = vars(ap.parse_args())

# Declare the number of epochs to train upon
epochUsed = 100

# Declare the learning rate
learningRate = 1e-4

# Declare the batch size to train upon
batchSize = 32

# Declaring variables to store the data and labels corresponding to that
info = []
labelsRequired = []

# Taking the path to images and shuffling the images randomly
Images = sorted(list(imutils.paths.list_images(args["images"])))
random.seed(53)
random.shuffle(Images)

# looping on the images taken as input
for path_to_image in Images:
    # Take in the image, load it, operate on it known as preprocessing and then store the processed image
    try:
        inputImage = cv2.imread(path_to_image)
        inputImage = cv2.resize(inputImage, (32, 32))
        inputImage = img_to_array(inputImage)
        info.append(inputImage)

        # The image path contains the label, extract the label and feed the label to the variable
        labelReceived = path_to_image.split(os.path.sep)[-2]
        labelReceived = 1 if labelReceived == "ambulance" else 0
        labelsRequired.append(labelReceived)
    except Exception as e:
        print(str(e))

# Normalizing or scaling the pixel values in the range of [0-1]
info = numpy.array(info, dtype="float") / 255.0
labelsRequired = numpy.array(labelsRequired)

# Divide or partition the data taken in into train and test datasets.
# Training dataset will contain 75% of the data and testing dataset will contain 25% of the data.
(X_train, X_test, Y_train, Y_test) = train_test_split(info, labelsRequired, test_size=0.25,
                                                                              random_state=53)

# Making the labels into vectors from integers
Y_train = to_categorical(Y_train, num_classes=2)
Y_test = to_categorical(Y_test, num_classes=2)

# construct the image generator for data augmentation
add = ImageDataGenerator(rotation_range=45, width_shift_range=0.2, height_shift_range=0.2,
                         zoom_range=0.1, horizontal_flip=True, shear_range=0.1,
                         fill_mode="nearest")

# Initializing the model with required parameters
print("....Compiling the model....")
modelBuild = build(width=32, height=32, depth=3, classes=2)
optimizer = Adam(lr=learningRate, decay=learningRate / epochUsed)
modelBuild.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# train the network
print("....training the network....")
H = modelBuild.fit_generator(add.flow(X_train, Y_train, batch_size=batchSize),
                             validation_data=(X_test, Y_test), steps_per_epoch=len(X_train) // batchSize,
                             epochs=epochUsed, verbose=1)

# Saving the build model to system
print("...Saving the model...")
modelBuild.save(args["modelToBeSaved"])

# Saving and plotting the validation and training loss
matplot.style.use("ggplot")
matplot.figure()
rangeEpoch = epochUsed
matplot.figure()
matplot.plot(numpy.arange(0, rangeEpoch), H.history["loss"], label="loss_in_training")
matplot.plot(numpy.arange(0, rangeEpoch), H.history["val_loss"], label="Loss_in_Validation")
matplot.title("Training Loss on Ambulance/Not Ambulance")
matplot.xlabel("Epoch #")
matplot.ylabel("Loss")
matplot.legend(loc="lower left")
matplot.savefig("Loss_32.png")

# Saving and plotting the validation and training accuracy
matplot.figure()
matplot.plot(numpy.arange(0, rangeEpoch), H.history["acc"], label="Accuracy_in_train")
matplot.plot(numpy.arange(0, rangeEpoch), H.history["val_acc"], label="Accuracy_in_validation")
matplot.title("Training Accuracy on Ambulance/Not Ambulance")
matplot.xlabel("Epoch #")
matplot.ylabel("Accuracy")
matplot.legend(loc="lower left")
matplot.savefig("Accuracy_32.png")