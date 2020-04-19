import tensorflow as tf

"""This script implements the functions for reading data.
"""

def load_data():
	"""Load the MNIST dataset and normalize pixel values into [0,1].

	Returns:
		x_train: An array of shape [60000, 784].
		y_train: An array of shape [60000,].
		x_test: An array of shape [10000, 784].
		y_test: An array of shape [10000,].
	"""
	# Load the dataset
	mnist = tf.keras.datasets.mnist
	(x_train, y_train),(x_test, y_test) = mnist.load_data()

	# Flatten: reshape images into a 1D array
	x_train = x_train.reshape((x_train.shape[0], -1))
	x_test = x_test.reshape((x_test.shape[0], -1))

	# Normalization: change pixel values from [0,255] to [0,1]
	x_train, x_test = x_train / 255.0, x_test / 255.0

	return x_train, y_train, x_test, y_test

def train_valid_split(x_train, y_train, split_index=50000):
	"""Split the original training data into a new training dataset
	and a validation dataset.

	Args:
		x_train: An array of shape [60000, 784].
		y_train: An array of shape [60000,].
		split_index: An integer.

	Returns:
		x_train_new: An array of shape [split_index, 784].
		y_train_new: An array of shape [split_index,].
		x_valid: An array of shape [60000-split_index, 784].
		y_valid: An array of shape [60000-split_index,].
	"""
	x_train_new = x_train[:split_index]
	y_train_new = y_train[:split_index]
	x_valid = x_train[split_index:]
	y_valid = y_train[split_index:]

	return x_train_new, y_train_new, x_valid, y_valid

