import tensorflow as tf
import numpy as np
import pickle, tqdm, os, time
'''
Homework2: Principal Component Analysis and Autoencoders

Useful Numpy functions
----------------------
In this assignment, you may want to use following helper functions:
- np.linalg.svd(): compute the singular value decomposition on a given matrix. 
- np.dot(): matrices multiplication operation.
- np.mean(): compute the mean value on a given matrix.
- np.zeros(): generate a all '0' matrix with a certain shape.
- np.expand_dims: expand the dimension of an array at the referred axis.
- np.squeeze: Remove single-dimensional entries from the shape of an array. 
- np.transpose(): matrix transpose operation.
- np.linalg.norm(): compute the norm value of a matrix. You may use it for the reconstruct_error function.

Tensorflow functions and APIs you may need
------------------------------------------
tf.Variable
tf.matmul
tf.transpose
tf.layers.dense
tf.nn.relu
tf.nn.sigmoid
tf.nn.tanh
'''

class PCA():
    '''
    Important!! Read before starting.
    1. To coordinate with the note at http://people.tamu.edu/~sji/classes/PCA.pdf,
    we set the input shape to be [256, n_samples].
    2. According to the note, input matrix X should be centered before doing SVD

    '''  
    def __init__(self, X, n_components):
        '''
        Args:
            X: The data matrix of shape [n_features, n_samples].
            n_components: The number of principal components. A scaler number.
        '''

        self.n_components = n_components
        self.X = X
        self.Up, self.Xp = self._do_pca()
    
    def _do_pca(self):
        '''
        To do PCA decomposition.
        Returns:
            Up: Principal components (transform matrix) of shape [n_features, n_components].
            Xp: The reduced data matrix after PCA of shape [n_components, n_samples].
        '''
        ### YOUR CODE HERE
        #subtract this from every column present in X
        self.X_mean = self.X.mean(axis=1, keepdims=True)
        X_centered = self.X - self.X_mean

        #Take the SVD of centered matrix
        U, S, Vh = np.linalg.svd(X_centered, full_matrices=True)
        #Take only the first n_components number of columns for Up
        Up = U[:, :self.n_components]
        #Reduced data matrix is given by
        Xp = np.dot(np.transpose(Up), X_centered)
        ### END YOUR CODE
        return Up, Xp

    def get_reduced(self, X=None):
        '''
        To return the reduced data matrix.
        Args:
            X: The data matrix with shape [n_features, n_any] or None. 
               If None, return reduced training X.
        Returns:
            Xp: The reduced data matrix of shape [n_components, n_any].
        '''
        if X is None: 
            return self.Xp, self.Up
        else:
            return self.Up.T @ X, self.Up

    def reconstruction(self, Xp):
        '''
        To reconstruct reduced data given principal components Up.

        Args:
        Xp: The reduced data matrix after PCA of shape [n_components, n_samples].

        Return:
        X_re: The reconstructed matrix of shape [n_features, n_samples].
        '''
        ### YOUR CODE HERE
        X_re = np.dot(self.Up, Xp)
        #Add back the mean to the reconstructed components
        X_re = X_re + self.X_mean
        ### END YOUR CODE
        return X_re


def frobeniu_norm_error(A, B):
    '''
    To compute Frobenius norm's square of the matrix A-B. It can serve as the
    reconstruction error between A and B, or can be used to compute the 
    difference between A and B.

    Args: 
    A & B: Two matrices needed to be compared with. Should be of same shape.

    Return: 
    error: the Frobenius norm's square of the matrix A-B. A scaler number.
    '''
    return np.linalg.norm(A-B)


class AE():
    '''
    Important!! Read before starting.
    1. To coordinate with the note at http://people.tamu.edu/~sji/classes/PCA.pdf and
    compare with PCA, we set the shape of input to the network as [256, n_samples].
    2. Do not do centering. Even though X in the note is the centered data, the neural network is 
    capable to learn this centering process. So unlike PCA, we don't center X for autoencoders,
    and we will still get the same results.
    3. Don't change or slightly change hyperparameters like learning rate, batch size, number of
    epochs for 5(e), 5(f) and 5(g). But for 5(h), you can try more hyperparameters and achieve as good results
    as you can.

    '''   
    def __init__(self, sess, d_hidden_rep):
        '''
        Args:
            sess: tf.Session. A Tensorflow session.
            d_hidden_rep: The dimension for the hidden representation in AE. A scaler number.
            n_features: The number of initial features, 256 for this dataset.
            
        Attributes:
            X: Tensor of shape [256, None]. A placeholder 
               for input images. "None" refers to any batch size.
            out_layer: Tensor of shape [256, None]. Output signal
               of network
            initializer: Initialize the trainable weights.
        '''
        self.sess = sess
        self.d_hidden_rep = d_hidden_rep
        self.n_features = 256
        self.X = tf.placeholder(tf.float32, shape=(self.n_features, None))
        self.w = tf.placeholder(tf.float32, shape=(self.n_features, self.d_hidden_rep))
        self.out_layer = self._network(self.X)
        
    def _network(self, X):
        '''

        You are free to use the listed functions and APIs from tf.layers or tf.nn:
            tf.Variable
            tf.matmul
            tf.transpose
            tf.layers.dense
            tf.nn.relu
            tf.nn.sigmoid
            tf.nn.tanh
        
        Args:
            X: Tensor. A placeholder of shape [n_features, None].
                for input images.
            You also need to define and initialize weights here.

        Returns:
            out_layer: Tensor of shape [n_features, None].
            
        '''
        initializer = tf.variance_scaling_initializer()
        
        ### YOUR CODE HERE

        '''
        Note: you should include all the three variants of the networks here. 
        You can comment the other two when you running one, but please include 
        and uncomment all the three in you final submissions.
        '''

        # Note: here for the network with weights sharing. Basically you need to follow the
        # formula (WW^TX) in the note at http://people.tamu.edu/~sji/classes/PCA.pdf .
        #W is of shape pxk, where p is dimension and k is reduced dimension
        self.w = tf.Variable(initializer([self.n_features, self.d_hidden_rep]), dtype=tf.float32)
        #activation_function = tf.nn.relu
        hidden_layer = tf.matmul(tf.transpose(self.w), self.X)
        self.out_layer = tf.matmul(self.w, hidden_layer)

        # Note: here for the network without weights sharing 
        print("########AE: Without weights sharing############")
        w_encoder = tf.Variable(initializer([self.n_features, self.d_hidden_rep]), dtype=tf.float32)
        w_decoder = tf.Variable(initializer([self.n_features, self.d_hidden_rep]), dtype=tf.float32)
        hidden_layer = tf.matmul(tf.transpose(w_encoder), self.X)
        self.out_layer = tf.matmul(w_decoder, hidden_layer)

        # Note: here for the network with more layers and nonlinear functions
        ####################### Network with 1 hidden layer and non linear activation functions##############
        self.d_hidden_rep = 64
        activation_function = tf.nn.relu
        w_encoder = tf.Variable(initializer([self.n_features, self.d_hidden_rep]), dtype=tf.float32)
        w_decoder = tf.Variable(initializer([self.n_features, self.d_hidden_rep]), dtype=tf.float32)
        hidden_layer = activation_function(tf.matmul(tf.transpose(w_encoder), self.X))
        self.out_layer = tf.nn.sigmoid(tf.matmul(w_decoder, hidden_layer))

        self.d_hidden_rep = 64
        activation_function = tf.nn.sigmoid
        w_encoder = tf.Variable(initializer([self.n_features, self.d_hidden_rep]), dtype=tf.float32)
        w_decoder = tf.Variable(initializer([self.n_features, self.d_hidden_rep]), dtype=tf.float32)
        hidden_layer = activation_function(tf.matmul(tf.transpose(w_encoder), self.X))
        self.out_layer = activation_function(tf.matmul(w_decoder, hidden_layer))

        self.d_hidden_rep = 64
        activation_function = tf.nn.tanh
        w_encoder = tf.Variable(initializer([self.n_features, self.d_hidden_rep]), dtype=tf.float32)
        w_decoder = tf.Variable(initializer([self.n_features, self.d_hidden_rep]), dtype=tf.float32)
        hidden_layer = activation_function(tf.matmul(tf.transpose(w_encoder), self.X))
        self.out_layer = tf.nn.sigmoid(tf.matmul(w_decoder, hidden_layer))

        #######################Network with 3 hidden layers and non linear activation functions###############
        self.d_hidden_rep = 64
        activation_function = tf.nn.relu
        w1 = tf.Variable(initializer([self.n_features, 128]), dtype=tf.float32)
        w2 = tf.Variable(initializer([128, self.d_hidden_rep]), dtype=tf.float32)
        w3 = tf.Variable(initializer([128, self.d_hidden_rep]), dtype=tf.float32)
        w4 = tf.Variable(initializer([self.n_features, 128]), dtype=tf.float32)

        hidden_layer_1 = activation_function(tf.matmul(tf.transpose(w1), self.X))
        hidden_layer_2 = activation_function(tf.matmul(tf.transpose(w2), hidden_layer_1))
        hidden_layer_3 = activation_function(tf.matmul(w3, hidden_layer_2))
        self.out_layer = tf.nn.sigmoid(tf.matmul(w4, hidden_layer_3))

        self.d_hidden_rep = 64
        activation_function = tf.nn.tanh
        w1 = tf.Variable(initializer([self.n_features, 128]), dtype=tf.float32)
        w2 = tf.Variable(initializer([128, self.d_hidden_rep]), dtype=tf.float32)
        w3 = tf.Variable(initializer([128, self.d_hidden_rep]), dtype=tf.float32)
        w4 = tf.Variable(initializer([self.n_features, 128]), dtype=tf.float32)

        hidden_layer_1 = activation_function(tf.matmul(tf.transpose(w1), self.X))
        hidden_layer_2 = activation_function(tf.matmul(tf.transpose(w2), hidden_layer_1))
        hidden_layer_3 = activation_function(tf.matmul(w3, hidden_layer_2))
        self.out_layer = tf.nn.sigmoid(tf.matmul(w4, hidden_layer_3))

        self.d_hidden_rep = 64
        activation_function = tf.nn.sigmoid
        w1 = tf.Variable(initializer([self.n_features, 128]), dtype=tf.float32)
        w2 = tf.Variable(initializer([128, self.d_hidden_rep]), dtype=tf.float32)
        w3 = tf.Variable(initializer([128, self.d_hidden_rep]), dtype=tf.float32)
        w4 = tf.Variable(initializer([self.n_features, 128]), dtype=tf.float32)

        hidden_layer_1 = activation_function(tf.matmul(tf.transpose(w1), self.X))
        hidden_layer_2 = activation_function(tf.matmul(tf.transpose(w2), hidden_layer_1))
        hidden_layer_3 = activation_function(tf.matmul(w3, hidden_layer_2))
        self.out_layer = tf.nn.sigmoid(tf.matmul(w4, hidden_layer_3))
        ### END YOUR CODE
        return self.out_layer
    
    
    def _setup(self):
        '''
        Model and training setup.
 
        Attributes:
            loss: Tensor of shape [1,]. Cross-entropy loss computed on
                the current batch.
            optimizer: tf.train.Optimizer. The optimizer for training
                the model. Different optimizers use different gradient
                descend policies.
            train_op: An Operation that updates the variables.
        '''
        self.loss = tf.reduce_mean(
            tf.square(self.out_layer-self.X))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
 
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.loss)

    
    def train(self, x_train, x_valid, batch_size, max_epoch):

        '''
        Autoencoder is an unsupervised learning method. To compare with PCA,
        it's ok to use the whole training data for validation and reconstruction.
        '''
 
        self._setup()
        self.sess.run(tf.global_variables_initializer())
 
        num_samples = x_train.shape[1]
        num_batches = int(num_samples / batch_size)
 
        num_valid_samples = x_valid.shape[1]
        num_valid_batches = (num_valid_samples - 1) // batch_size + 1

        print('---Run...')
        for epoch in range(1, max_epoch + 1):
 
            # To shuffle the data at the beginning of each epoch.
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[:, shuffle_index]
 
            # To start training at current epoch.
            loss_value = []
            qbar = tqdm.tqdm(range(num_batches))
            for i in qbar:
                batch_start_time = time.time()
 
                start = batch_size * i
                end = batch_size * (i + 1)
                x_batch = curr_x_train[:, start:end]

                feed_dict = {self.X: x_batch}
                loss, _ = self.sess.run(
                    [self.loss, self.train_op],
                    feed_dict=feed_dict)
                if not i % 10:
                    qbar.set_description(
                        'Epoch {:d} Loss {:.6f}'.format(
                            epoch, loss))
 
            # To start validation at the end of each epoch.
            loss = 0
            print('Doing validation...', end=' ')
            for i in range(num_valid_batches):
                start = batch_size * i
                end = min(batch_size * (i + 1), x_valid.shape[1])
                x_valid_batch = x_valid[:, start:end]
 
                feed_dict = {self.X: x_valid_batch}
                loss += self.sess.run(self.loss, feed_dict=feed_dict)
 
            print('Validation Loss {:.6f}'.format(loss))
 

    def get_params(self):
        """
        Get parameters for the trained model.
        
        Returns:
            final_w: An array of shape [n_features, d_hidden_rep].
        """
        final_w = self.sess.run(self.w)
        return final_w
    
    def reconstruction(self, X):
        '''
        To reconstruct data. You’re required to reconstruct one by one here,
        that is to say, for one loop, input to the network is of the shape [n_features, 1].
        Args:
            X: The data matrix with shape [n_features, n_any]
        Returns:
            X_re: The reconstructed data matrix, which has the same shape as X.
        '''
        _, n_samples = X.shape
        n_features = X.shape[0]
        X_re = np.empty([n_features, n_samples])
        for i in range(n_samples): 
            ### YOUR CODE HERE

            # Note: Format input curr_X to the shape [n_features, 1]
            curr_X = X[:, i].reshape(n_features, 1)
            ### END YOUR CODE            
            feed_dict={
                self.X: np.array(curr_X)}
            curr_recons = self.sess.run(self.out_layer, feed_dict=feed_dict)
            ### YOUR CODE HERE
            # Note: To achieve final reconstructed data matrix with the shape [n_features, n_any].
            X_re[:, i] = curr_recons.reshape(n_features)
            ### END YOUR CODE 
        return X_re