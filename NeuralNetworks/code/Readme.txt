Instruction of programming assignments for CSCE636: Deep Learning


We will use the Python programming language and Tensorflow for this assignment.


Installation of Python
-----------------------
For all assignments in this course. Specifically, we will use a few popular libraries (numpy, matplotlib, math) for scientific computing. We expect that many of you already have some basic experience with Python and Numpy. We also provide basic guide to learn Python and Numpy.

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
You may need to install the GPU version if running code on GPU.


Dataset Descriptions
--------------------
We will use MNIST dataset, which is much larger than the USPS dataset.
You can find the description of the dataset here:
http://yann.lecun.com/exdb/mnist/


Assignment Descriptions
-----------------------
This assignment is designed to help you get familiar with Tensorflow. To help you understand all necessary parts, only basic Tensorflow APIs in tf.layers and tf.nn are allowed to use. Most parts of the code are given. Please make sure that you understand how the code works for future assignments.

``DataReader.py'' implements the functions for reading data. Please read and completely understand how to read and process data before implementing other parts.

In ``MNIST\_tutorial.py'', an implementation using advanced APIs is provided. You can use it to check your results. A link to Tensorflow turorials is also provided.

Only write your code between the following lines. Do not modify other parts. Only the code
between the following lines will be graded. Don NOT include experimental results in your
code submission. Do NOT change file names.
### YOUR CODE HERE
### END YOUR CODE


Refer to https://www.tensorflow.org/api_docs/python/tf for more details.

Feel free to email Yi Liu for any assistance.
Email address: yiliu@tamu.edu.




