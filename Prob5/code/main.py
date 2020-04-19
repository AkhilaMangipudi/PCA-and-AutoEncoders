import tensorflow as tf
from helper import load_data
from solution import PCA, AE, frobeniu_norm_error
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '8'


def test_pca(p):
    pca = PCA(A, p)
    Ap, G = pca.get_reduced()
    A_re = pca.reconstruction(Ap)
    error = frobeniu_norm_error(A, A_re)
    print('PCA-Reconstruction error for {k} components is'.format(k=p), error)
    return G

def test_ae(p):
    model = AE(sess=tf.Session(), d_hidden_rep=p)
    model.train(A, A, 128, 300)
    A_re = model.reconstruction(A)
    final_w = model.get_params()
    error = frobeniu_norm_error(A, A_re)
    print('AE-Reconstruction error for {k}-dimensional hidden representation is'.format(k=p), error)
    return final_w

if __name__ == '__main__':
    dataloc = "./data/USPS.mat"
    A = load_data(dataloc)
    A = A.T
    ## Normalize A
    A = A/A.max()

    ### YOUR CODE HERE
    # Note: You are free to modify your code here for debugging and justifying your ideas for 5(f)

    ps = [32, 64, 128]
    for p in ps:
        G = test_pca(p)
        final_w = test_ae(p)
        ####### For question 5(f) ########
        matrix_transform_G = np.dot(G, G.T)
        matrix_transform_W = np.dot(final_w, final_w.T)
        matrix_error = frobeniu_norm_error(matrix_transform_G, matrix_transform_W)
        print('Error before transformation of G and W for {} components is {}'.format(p, frobeniu_norm_error(G, final_w)))
        print('Error after transformations of G and W for {} components is {}'.format(p, matrix_error))

    ### END YOUR CODE 

    

