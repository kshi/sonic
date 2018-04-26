import tensorflow as tf
import numpy as np
from utils import *

class Baseline(object):
    def __init__(self, sess, optimizer, ob_space, reuse=True):
        nh, nw, nc = ob_space.shape
        ob_shape = [None, nh, nw, nc]
        init_scale=1.0
        init_bias=0.0
        eps = 0.1
        
        X = tf.placeholder(tf.uint8, ob_shape)
        value_estimates = tf.placeholder(tf.float32, [None])
        scaled_images = tf.cast(X, tf.float32) / 255.
        activ = tf.nn.relu
        X = tf.placeholder(tf.uint8, ob_shape)
        scaled_images = tf.cast(X, tf.float32) / 255.
        h = activ(conv(scaled_images, 'v1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2)))
        h2 = activ(conv(h, 'v2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2)))
        h3 = activ(conv(h2, 'v3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2)))
        h4 = conv_to_fc(h3)
        vf = fc(h4, 'val', 1)[:,0]
        loss = tf.reduce_mean(tf.square(vf - value_estimates))
        self.train_op = optimizer.apply_gradients(optimizer.compute_gradients(loss))

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
