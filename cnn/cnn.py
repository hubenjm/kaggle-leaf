import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt

from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import img_to_array, load_img

data_path = "../"
image_path = "../images/"

# Swiss army knife function to organize the provided numeric data
def encode_numeric(train_data, test_data):
	le = preprocessing.LabelEncoder().fit(train_data.species) 
	labels = le.transform(train_data.species)           # encode species strings
	classes = list(le.classes_)                    # save column names for submission
	test_ids = test_data.pop('id')                             # save test_data ids for submission
	train_data = train_data.drop(['species', 'id'], axis=1)

	train_data = preprocessing.StandardScaler().fit(train_data).transform(train_data)
	test_data = preprocessing.StandardScaler().fit(test_data).transform(test_data)

	return train_data, labels, test_data, test_ids, classes

def resize_img(img, max_dim=96):
    """
    Resize the image to so that the maximum side is of size max_dim
    Returns a new image of the right size
    """
    # Get the axis with the larger dimension
    max_ax = max((0, 1), key=lambda i: img.size[i])
    # Scale both axes so the image's largest dimension is max_dim
    scale = max_dim / float(img.size[max_ax])
    return img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))

def load_image_data(ids, max_dim=96, center=True):
    """
    Takes as input an array of image ids and loads the images as numpy
    arrays with the images resized so that the longest side is max-dim length.
    If center is True, then will place the image in the center of
    the output array, otherwise it will be placed at the top-left corner.
    """
    # Initialize the uniformly-sized output array
    X = np.empty((len(ids), max_dim, max_dim, 1))

    for i, ids_i in enumerate(ids):
        # Turn the image into an array
        x = resize_img(load_img(image_path + str(ids_i) + ".jpg", grayscale=True), max_dim=max_dim)
        x = img_to_array(x)
        # Get the corners of the bounding box for the image
        length = x.shape[0]
        width = x.shape[1]
        if center:
            h1 = int((max_dim - length) / 2)
            h2 = h1 + length
            w1 = int((max_dim - width) / 2)
            w2 = w1 + width
        else:
            h1, w1 = 0, 0
            h2, w2 = (length, width)
        # Insert into image matrix
        X[i, h1:h2, w1:w2, 0:1] = x
    # Scale the array values so they are between 0 and 1
    return np.around(X / 255.0)



def main():

	

