import numpy as np
import sys
import os
import tensorflow as tf

###################################################################
# Variables                                                       #
# When launching project or scripts from Visual Studio,           #
# input_dir and output_dir are passed as arguments automatically. #
# Users could set them from the project setting page.             #
###################################################################

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("input_dir", ".", "Input directory where training dataset and meta data are saved")
tf.app.flags.DEFINE_string("output_dir", ".", "Output directory where output such as logs are saved.")
tf.app.flags.DEFINE_string("log_dir", ".", "Model directory where final model files are saved.")

def main(_):
    # TODO: add your code here


    with tf.Session() as sess:
        welcome = sess.run(tf.constant("Hello, TensorFlow!"))
        print(welcome)
      
        #a = tf.constant(2.0, tf.float32)
        #b=tf.constant(3.0)
        #print(a,b)

        a = tf.placeholder(tf.float32)
        b=a*2
        with tf.Session() as sess:
            result = sess.run(b, feed_dict={a:23.0})
            print (result)
    exit(0)


if __name__ == "__main__":
    tf.app.run()
