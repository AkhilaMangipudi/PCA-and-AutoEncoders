import os
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from Network import MLP

"""This script defines the training, validation and testing process.
"""

class MNIST(object):

	def __init__(self, sess, conf):
		self.sess = sess
		self.conf = conf

	def setup(self, training):
		print('---Setup input interfaces...')
		self.inputs = tf.placeholder(tf.float64, shape=(None, 784))
		self.labels = tf.placeholder(tf.int32)

		print('---Setup the network...')
		network = MLP(self.conf.num_hid_layers,
						self.conf.num_hid_units,
						self.conf.num_classes)

		if training:
			print('---Setup training components...')
			# compute logits
			logits = network(self.inputs, True)

			# predictions for validation
			self.preds = tf.argmax(logits, axis=-1)

			# loss function
			self.losses = tf.losses.sparse_softmax_cross_entropy(
				self.labels, logits, reduction=tf.losses.Reduction.MEAN)

			# optimizer - Adam
			optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
			self.train_op = optimizer.minimize(self.losses)

			print('---Setup the Saver for saving models...')
			self.saver = tf.train.Saver(var_list=tf.global_variables(),
										max_to_keep=0)

		else:
			print('---Setup testing components...')
			# compute predictions
			logits = network(self.inputs, False)
			self.preds = tf.argmax(logits, axis=-1)

			print('---Setup the Saver for loading models...')
			self.loader = tf.train.Saver(var_list=tf.global_variables())


	def train(self, x_train, y_train, x_valid, y_valid,
								max_epoch, validation=True):
		print('###Train###')

		self.setup(True)
		self.sess.run(tf.global_variables_initializer())

		# Determine how many batches in an epoch
		num_samples = x_train.shape[0]
		num_batches = int(num_samples / self.conf.batch_size)
		
		print('---Run...')
		for epoch in range(1, max_epoch+1):

			start_time = time.time()
			# Shuffle
			shuffle_index = np.random.permutation(num_samples)
			curr_x_train = x_train[shuffle_index]
			curr_y_train = y_train[shuffle_index]

			loss_value = []
			for i in range(num_batches):
				# Current batch
				x_batch = curr_x_train\
						[self.conf.batch_size*i:self.conf.batch_size*(i+1)]
				y_batch = curr_y_train\
						[self.conf.batch_size*i:self.conf.batch_size*(i+1)]

				# Run
				feed_dict = {self.inputs: x_batch,
							self.labels: y_batch}
				loss, _ = self.sess.run(
							[self.losses, self.train_op],
							feed_dict=feed_dict)
				print('Batch {:d}/{:d} Loss {:.6f}'.format(
						i, num_batches, loss),
						end='\r',
						flush=True)
			duration = time.time() - start_time
			print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(
						epoch, loss, duration))
			
			if validation and x_valid is not None and y_valid is not None:
				print('###Validation###')

				### YOUR CODE HERE
				
				# Note: implement the validation process.
				# Hint: read the testing process.
				preds = []
				for i in tqdm(range(x_valid.shape[0])):
					feed_dict = {self.inputs: x_valid[i].reshape((1,-1)), self.labels: y_valid[i]}
					preds.append(self.sess.run(self.preds, feed_dict=feed_dict))

				preds = np.array(preds).reshape(y_valid.shape)
				print('Validation accuracy: {:.4f}'.format(np.sum(preds == y_valid) / y_valid.shape[0]))
				
				### END CODE HERE

		if not validation:
			self.save(self.saver, epoch)

	def test(self, x_test, y_test, checkpoint_num):
		print('###Test###')

		self.setup(False)
		self.sess.run(tf.global_variables_initializer())

		# load checkpoint
		checkpointfile = self.conf.modeldir+'/model.ckpt-'+str(checkpoint_num)
		self.load(self.loader, checkpointfile)

		preds = []
		for i in tqdm(range(x_test.shape[0])):
			feed_dict = {self.inputs: x_test[i].reshape((1,-1)),
						self.labels: y_test[i]}
			preds.append(self.sess.run(self.preds,
							feed_dict=feed_dict))

		preds = np.array(preds).reshape(y_test.shape)
		print('Test accuracy: {:.4f}'.format(
							np.sum(preds==y_test)/y_test.shape[0]))


	def save(self, saver, step):
		'''Save weights.
		'''
		model_name = 'model.ckpt'
		checkpoint_path = os.path.join(self.conf.modeldir, model_name)
		if not os.path.exists(self.conf.modeldir):
			os.makedirs(self.conf.modeldir)
		saver.save(self.sess, checkpoint_path, global_step=step)
		print('The checkpoint has been created.')

	def load(self, loader, filename):
		'''Load trained weights.
		''' 
		loader.restore(self.sess, filename)
		print("Restored model parameters from {}".format(filename))













