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

from digit_utils import load_data

from train import load_model_from_file

data = load_data("../dessin.png")
new_shape = tuple([1] + list(data.shape) + [1])
data=data.reshape(new_shape)
print(new_shape, data.shape)

model = load_model_from_file()
pred = model.predict(data)
print(pred)