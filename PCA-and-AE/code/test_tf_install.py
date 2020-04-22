import tensorflow as tf 

hello = tf.constant("hello TensorFlow!") 
sess=tf.Session() 
print(sess.run(hello))
print("Version", tf.__version__)