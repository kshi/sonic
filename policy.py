import tensorflow as tf
import numpy as np
from utils import *

class Policy(object):
    def __init__(self, sess, optimizer, ob_space, ac_space):
        nh, nw, nc = ob_space.shape
        ob_shape = [None, nh, nw, nc]
        nact = 8
        init_scale=1.0
        init_bias=0.0
        eps = 0.1
        
        X = tf.placeholder(tf.uint8, ob_shape)
        scaled_images = tf.cast(X, tf.float32) / 255.

        h = tf.layers.conv2d(scaled_images, 32, 8, strides=4, activation=tf.nn.relu, kernel_initializer=tf.orthogonal_initializer(np.sqrt(2)))
        h2 = tf.layers.conv2d(scaled_images, 64, 4, strides=2, activation=tf.nn.relu, kernel_initializer=tf.orthogonal_initializer(np.sqrt(2)))
        h3 = tf.layers.conv2d(scaled_images, 64, 3, strides=1, activation=tf.nn.relu, kernel_initializer=tf.orthogonal_initializer(np.sqrt(2)))
        h4 = tf.layers.flatten(h3)
        pi = tf.layers.dense(h4, nact, activation=None, kernel_initializer=tf.orthogonal_initializer(np.sqrt(2)))
        prob = tf.nn.softmax(pi)
        
        Q_estimate = tf.placeholder(tf.float32, [None])
        actions = tf.placeholder(tf.int32, [None])
        old_prob = tf.placeholder(tf.float32, [None])
        
        indices = tf.range(0, tf.shape(actions)[0])
        slice_indices = tf.stack([indices, actions], axis=1)
        action_prob = tf.gather_nd(prob, slice_indices)

        ratio = tf.divide(action_prob, old_prob)
        pg_obj = tf.multiply(ratio, Q_estimate)
        clip_obj = tf.multiply(tf.clip_by_value(ratio, 1-eps, 1+eps), Q_estimate)
        surrogate_loss = -tf.reduce_mean(tf.minimum(pg_obj, clip_obj))
        
        self.train_op = optimizer.minimize(surrogate_loss)

        self.state = X
        self.prob = prob
        self.old_prob = old_prob
        self.actions = actions
        self.Q_estimate = Q_estimate
        self.loss = surrogate_loss
        self.optimizer = optimizer
        self.sess = sess 
        
    def compute_prob(self, states):
        return self.sess.run(self.prob, feed_dict={self.state:states})
    
    def compute_val(self, states):
        return self.sess.run(self.val, feed_dict={self.state:states})
    
    def compute_prob_act(self, states, actions):
        prob = self.sess.run(self.prob, feed_dict={self.state:states})
        indices = tf.range(0, tf.shape(actions)[0])
        slice_indices = tf.stack([indices, actions], axis=1)
        return self.sess.run(tf.gather_nd(prob, slice_indices))

    def train(self, states, actions, Qs, old_prob):
        self.sess.run(self.train_op, feed_dict={self.state:states, self.actions:actions, self.Q_estimate:Qs, self.old_prob:old_prob})
