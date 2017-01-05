import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt

from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator, array_to_img
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten, Input, merge
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

num_data_path = "../"
image_path = "../images/"

# function to organize/clean the provided numeric data
def encode_numeric_data(train_data, test_data):
	le = preprocessing.LabelEncoder().fit(train_data.species) 
	labels = le.transform(train_data.species)           # encode species strings
	classes = list(le.classes_)   	                # save column names for submission
	train_ids = train_data.pop('id')	
	test_ids = test_data.pop('id')                            # save test_data ids for submission
	train_data = train_data.drop(['species'], axis=1)

	train_data = preprocessing.StandardScaler().fit(train_data).transform(train_data)
	test_data = preprocessing.StandardScaler().fit(test_data).transform(test_data)

	return train_data, train_ids, labels, test_data, test_ids, classes

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

#define modified NumpyArrayIterator and ImageDateGenerator classes (taken from https://www.kaggle.com/abhmul/leaf-classification/keras-convnet-lb-0-0052-w-visualization)
#the key here is to make the index_array used in the next() method of the NumpyArrayIterator a class variable, so that it can be accessed externally to be able
#to access the corresponding row from the pre-extracted feature data

class MyImageDataGenerator(ImageDataGenerator):
	def flow(self, X, y=None, batch_size=32, shuffle=True, seed=None,
			save_to_dir=None, save_prefix='', save_format='jpeg'):
		return MyNumpyArrayIterator(
			X, y, self,
			batch_size=batch_size, shuffle=shuffle, seed=seed,
			dim_ordering=self.dim_ordering,
			save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)


class MyNumpyArrayIterator(NumpyArrayIterator):
	def next(self):
		# for python 2.x.
		# Keeps under lock only the mechanism which advances
		# the indexing of each batch
		# see http://anandology.com/blog/using-iterators-and-generators/
		with self.lock:
			# We changed index_array to self.index_array
			self.index_array, current_index, current_batch_size = next(self.index_generator)
		# The transformation of images is not under thread lock so it can be done in parallel
		batch_x = np.zeros(tuple([current_batch_size] + list(self.X.shape)[1:]))
		for i, j in enumerate(self.index_array):
			x = self.X[j]
			x = self.image_data_generator.random_transform(x.astype('float32'))
			x = self.image_data_generator.standardize(x)
			batch_x[i] = x
		if self.save_to_dir:
			for i in range(current_batch_size):
				img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
				fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
																index=current_index + i,
																hash=np.random.randint(1e4),
																format=self.save_format)
				img.save(os.path.join(self.save_to_dir, fname))
		if self.y is None:
			return batch_x
		batch_y = self.y[self.index_array]
		return batch_x, batch_y

def merged_model(dim):
	#combine the convolutional neural network with image data input with the pre-extracted feature data
	#then run through a fully-connected layer(s) with relu activation 
	#can mess with layer settings explicitly below as desired.

	# Define the image input
	image = Input(shape=(dim, dim, 1), name='image')

	# Pass it through the first convolutional layer
	x = Convolution2D(8, 5, 5, input_shape=(dim, dim, 1), border_mode='same')(image)
	x = (Activation('relu'))(x)
	x = (MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(x)

	# Now through the second convolutional layer
	x = (Convolution2D(32, 5, 5, border_mode='same'))(x)
	x = (Activation('relu'))(x)
	x = (MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(x)

	# Flatten our array
	x = Flatten()(x)

	# Define the pre-extracted feature input
	numerical = Input(shape=(192,), name='numerical') #the 192 is crucial as it depends explicitly on the structure of the pre-extracted feature data
	# Concatenate the output of CNN with the pre-extracted feature input
	concatenated = merge([x, numerical], mode='concat')

	# Add a fully connected layer
	x = Dense(200, activation='relu')(concatenated)
	x = Dropout(.5)(x)

	# Add a second fully connected layer
	x = Dense(100, activation='relu')(concatenated)

	# Get the final output
	out = Dense(99, activation='softmax')(x) #99 is crucial as this is the number of actual unique species in the data set
	# create the model using the Keras functional API
	model = Model(input=[image, numerical], output=out)
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy']) #can change optimizer settings as desired

	return model

def combined_generator(image_generator, X_num):
	"""
	A generator to train our keras neural network. It
	takes the image augmenter generator and the array
	of the pre-extracted features.
	It yields a minibatch and will run indefinitely
	"""
	while True:
		for i in range(X_num.shape[0]):
			# Get the image batch and labels
			batch_image, batch_y = next(image_generator)
			# This is where that change to the source code we
			# made will come in handy. We can now access the indices
			# of the images that image_generator gave us.
			x = X_num[image_generator.index_array]
			yield [batch_image, x], batch_y

def main():

	max_image_dim = 128 #number of pixels in each dimension of image
	split = 0.1
	seed = 2017

	#load numeric training data
	num_train_data = pd.read_csv(num_data_path + "train.csv")
	num_test_data = pd.read_csv(num_data_path + "test.csv")
	num_train_data, train_ids, labels, num_test_data, test_ids, classes = encode_numeric_data(num_train_data, num_test_data)
	
	#load image data
	image_train_data = load_image_data(train_ids, max_dim=max_image_dim, center=True)
	image_test_data = load_image_data(test_ids, max_dim=max_image_dim, center=True)

	#split training data into training and validation subsets using stratified shuffle split
	sss = StratifiedShuffleSplit(1, test_size=split, random_state=seed)
 	train_index, val_index = next(sss.split(num_train_data, labels))
	X_num_train, X_num_val = num_train_data[train_index], num_train_data[val_index]
	X_image_train, X_image_val = image_train_data[train_index], image_train_data[val_index]
	y_train, y_val = labels[train_index], labels[val_index]

	y_train_cat = to_categorical(y_train) #convert to binary vector for use with keras model
	y_val_cat = to_categorical(y_val)

	#create image data generator for batch learning
	image_generator = MyImageDataGenerator(
		rotation_range=20,
		zoom_range=0.2,
		horizontal_flip=True,
		vertical_flip=True,
		fill_mode='nearest')

	image_generator_train = image_generator.flow(X_image_train, y_train_cat, seed=seed)
	
	#create model
	model = merged_model(max_image_dim)

	#train model
	#autosave best model
	best_model_file = "leaf_predictor_nn.h5"
	best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1, save_best_only=True)

	history = model.fit_generator(combined_generator(image_generator_train, X_num_train),
                              samples_per_epoch=X_num_train.shape[0],
                              nb_epoch=100,
                              validation_data=([X_image_val, X_num_val], y_val_cat),
                              nb_val_samples=X_num_val.shape[0],
                              verbose=0,
                              callbacks=[best_model])

	print('Loading the best model...')
	model = load_model(best_model_file)
	print('Best Model loaded!')

	#compute predictions on test set and create submission
	#get predictions
	y_predict_prob = model.predict([image_test_data, num_test_data])

	# Get the names of the column headers
	text_labels = sorted(pd.read_csv(num_data_path + "train.csv").species.unique())

	## Converting the test predictions in a dataframe as depicted by sample submission
	y_submission = pd.DataFrame(y_predict_prob, index=test_ids, columns=text_labels)

	fp = open('submission.csv', 'w')
	fp.write(y_submission.to_csv())	
	print('Finished writing submission')
	## Display the submission
	y_submission.tail()

if __name__ == "__main__":
	main()
