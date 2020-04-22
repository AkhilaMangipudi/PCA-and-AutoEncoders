Instructions for programming assignments of CSCE636: Deep Learning


You will use the Python programming language and Tensorflow for this assignment.


Installation of Python
-----------------------
For all assignments in this course, we will use a few popular libraries (numpy, matplotlib, math) for scientific computing. 
We expect that many of you already have some basic experience with Python and Numpy. We also provide basic guide to learn Python and Numpy.

Download and install Anaconda with Python3.6 version:
- Download at the website: https://www.anaconda.com/download/
- Install Python3.6 version(not Python 2.7)
Anaconda will include all the Python libraries we need. 


Python & Numpy Tutorial
-----------------------
- Official Python tutorial: https://docs.python.org/3/tutorial/
- Official Numpy tutorial: https://docs.scipy.org/doc/numpy-dev/user/quickstart.html
- Good tutorial sources: http://cs231n.github.io/python-numpy-tutorial/ 


Installation of tqdm
---------------------
Please install python packages tqdm using:
"pip install tqdm" or
"conda install tqdm"


Installation of Tensorflow
--------------------------
If you are using Anaconda environment, simply using:
"conda install tensorflow"

For other environments, using:
"pip install tensorflow"

You can refer to https://www.tensorflow.org/install/pip for more details.

Note: by default, you are installing the CPU version of tensorflow.
You may need to install the GPU version if running code on the GPU.

After you finish the installation, run "test_tf_install.py" to test if
the installation is successful. If it is shown
'''
hello Tensoflow!
Version 1.13.1
'''
then you are all set. (1.13 is the recommanded version for tensorflow.)


Dataset Descriptions
--------------------
We will use USPS dataset for this assignments. The USPS dataset is in the “data” folder: USPS.mat. 
The whole data has already been loaded into the matrix A. The matrix A contains all the images of size 16 × 16. 
Each of the 3000 rows in A corresponds to the image of one handwritten digit (between 0 and 9). 


Assignment Descriptions
-----------------------
There are total four Python files including 'main.py', 'solution.py', 'helper.py' and 'test_tf.install.py'. 
In this assignment, you need to implement your solution in 'solution.py' and 'main.py' files following the given instruction. 
However, you might need to read all the files to fully understand the requirement. 

The 'helper.py' includes all the helper functions for the assignments, like load data, show images, etc.
The 'test_tf.install.py' is used to test if you successfully install the tensorflow.

Only implement your code in 'solution.py' and 'main.py' files.
Only write your code between the following lines. Do not modify other parts. Only the code
between the following lines will be graded. Don NOT include experimental results in your
code submission. Do NOT change file names.
### YOUR CODE HERE
### END YOUR CODE


Useful Numpy functions
----------------------
In this assignment, you mayuse following numpy functions:
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

Refer to https://www.tensorflow.org/api_docs/python/tf for more details.

Feel free to email Yi Liu for any assistance.
Email address: yiliu@tamu.edu.




