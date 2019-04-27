import pickle

import os

import scipy
import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, ZeroPadding2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model, Sequential, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from sklearn.preprocessing import LabelEncoder

from keras.utils import to_categorical

import keras.backend as K

K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from digit_utils import load_written_dataset
    
def get_test_model(input_shape):
    X_input = Input(input_shape)

    X = Conv2D(6, (2, 2), name='conv0')(X_input)

    X = Flatten()(X)
    X = Dense(10, activation='softmax', name='fc')(X)

    model =Model(inputs=X_input,outputs=X,name='TestModel')
    return model

def get_model(input_shape):
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    print("Input shape:", input_shape)
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((2, 2))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (8, 8), strides=(2, 2), name='conv0')(X)
    X = BatchNormalization(axis=2, name='bn0')(X)
    X = Activation('relu')(X)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(16, (6, 6), strides=(1, 1), name='conv1')(X)
    X = BatchNormalization(axis=2, name='bn1')(X)
    X = Activation('relu')(X)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(8, (4, 4), strides=(1, 1), name='conv2')(X)
    X = BatchNormalization(axis=2, name='bn2')(X)
    X = Activation('relu')(X)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(4, (2, 2), strides=(1, 1), name='conv3')(X)
    X = BatchNormalization(axis=2, name='bn3')(X)
    X = Activation('relu')(X)

    # # MAXPOOL
    # X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(10, activation='softmax', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs=X_input, outputs=X, name='DigitsModel')

    return model

def get_model_2(input_shape):
    # building a linear stack of layers with the sequential model
    model = Sequential()
    model.add(Dense(256, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model

def save_model(m):
    model.save("model.h5")

def save_test(x, y):
    with open("test.pckl", 'wb') as f:
        pickle.dump({
            "x":x,
            "y":y
        }, f)

def load_model_from_file():
    if os.path.isfile("model.h5"):
        m = load_model("model.h5")
        return m

    raise Exception("Couldn't find 'model.h5'")

def get_dataset():
    print("Loading")
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_written_dataset()

    print("Normalizing")
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.
    # We have 1 layer (B&W) so we need to say it
    dest_shape_train = tuple(list(X_train.shape) + [1])
    dest_shape_test = tuple(list(X_test.shape) + [1])
    X_train = X_train.reshape(dest_shape_train)
    X_test = X_test.reshape(dest_shape_test)

    Y_train = Y_train_orig
    Y_test = Y_test_orig

    # Flatten to make [[x], [y], ...] into [x, y, ...]
    Y_train = np.ndarray.flatten(Y_train)
    Y_test = np.ndarray.flatten(Y_test)
    # Encode to labels (integers while we currently have floats)
    encoder = LabelEncoder()
    encoder.fit(Y_train)
    Y_train_encoded = encoder.transform(Y_train)
    # Use one hot encoding
    Y_train_onehot = to_categorical(Y_train_encoded)

    encoder = LabelEncoder()
    encoder.fit(Y_test)
    Y_test_encoded = encoder.transform(Y_test)
    Y_test_onehot = to_categorical(Y_test_encoded)

    return X_train, X_test, Y_train_onehot, Y_test_onehot

debug=False

def create_and_train_model(ctor):
    X_train, X_test, Y_train, Y_test = get_dataset()

    shape = X_train.shape
    input_shape = tuple(list(X_train[0].shape))
    print("Input shape",input_shape)
    model = ctor(input_shape)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    if debug:
        model.fit(x=X_train, y=Y_train, epochs=1, batch_size=50)
    else:
        model.fit(x=X_train, y=Y_train, epochs=15, batch_size=50)

    return model, X_test, Y_test

if __name__ == '__main__':
    if debug:
        print("----------- DEBUG MODE --------------")
        constructor = get_test_model
    else:
        constructor = get_model 

    if os.path.isfile("model.h5"):
        print("Loading model from H5")
        model = load_model_from_file()
        _, X_test, _, Y_test = get_dataset()
    else:
        print("Creating model")
        model, X_test, Y_test = create_and_train_model(constructor)
        save_model(model)
        save_test(X_test, Y_test)

    preds = model.predict(X_test)
    print(preds)
    print()
    print("Loss = " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]))
