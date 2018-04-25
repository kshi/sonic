import tensorflow as tf
import numpy as np
from utils import *

class Baseline(object):
    def __init__(self, sess, optimizer, ob_space):
        nh, nw, nc = ob_space.shape
        ob_shape = [None, nh, nw, nc]
        init_scale=1.0
        init_bias=0.0
        eps = 0.1
        
        X = tf.placeholder(tf.uint8, ob_shape)
        value_estimates = tf.placeholder(tf.float32, [None])
        scaled_images = tf.cast(X, tf.float32) / 255.

        h = tf.layers.conv2d(scaled_images, 32, 8, strides=4, activation=tf.nn.relu, kernel_initializer=tf.orthogonal_initializer(np.sqrt(2)))
        h2 = tf.layers.conv2d(scaled_images, 64, 4, strides=2, activation=tf.nn.relu, kernel_initializer=tf.orthogonal_initializer(np.sqrt(2)))
        h3 = tf.layers.conv2d(scaled_images, 64, 3, strides=1, activation=tf.nn.relu, kernel_initializer=tf.orthogonal_initializer(np.sqrt(2)))
        h4 = tf.layers.flatten(h3)
        vf = tf.layers.dense(h4, 1, activation=None, kernel_initializer=tf.orthogonal_initializer(np.sqrt(2)))
        
        loss = tf.reduce_mean(tf.square(vf - value_estimates))
        self.train_op = optimizer.minimize(loss)

        self.state = X
        self.value_estimates = value_estimates
        self.val = vf
        self.loss = loss
        self.optimizer = optimizer
        self.sess = sess 
        
    def compute_val(self, states):
        return self.sess.run(self.val, feed_dict={self.state:states})
    
        indices = tf.range(0, tf.shape(actions)[0])
        slice_indices = tf.stack([indices, actions], axis=1)
        return self.sess.run(tf.gather_nd(prob, slice_indices))

    def train(self, states, targets):
        self.sess.run(self.train_op, feed_dict={self.state:states, self.value_estimates:targets})
