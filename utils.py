import numpy as np
import tensorflow as tf
def action_map(a):
    LEFT   =  (0,0,0,0,0,0,1,0,0,0,0,0)
    RIGHT  =  (0,0,0,0,0,0,0,1,0,0,0,0)
    DOWN   =  (0,0,0,0,0,1,0,0,0,0,0,0)
    JUMP   =  (1,0,0,0,0,0,0,0,0,0,0,0)
    R_DOWN =  (0,0,0,0,0,1,0,1,0,0,0,0)
    L_DOWN =  (0,0,0,0,0,1,1,0,0,0,0,0)
    NONE   =  (0,0,0,0,0,0,0,0,0,0,0,0)
    J_DOWN =  (1,0,0,0,0,1,0,0,0,0,0,0)
    mapping = {0:LEFT, 1:RIGHT, 2:DOWN, 3:R_DOWN, 4:L_DOWN, 5:J_DOWN, 6:JUMP, 7:NONE}
    return mapping[a]
def discounted_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_sum = 0
    for i in reversed(range(0,len(r))):
        discounted_r[i] = running_sum * gamma + r[i]
        running_sum = discounted_r[i]
    return list(discounted_r)
def conv(x, scope, *, nf, rf, stride, pad='VALID', init_scale=1.0, data_format='NHWC'):
    if data_format == 'NHWC':
        channel_ax = 3
        strides = [1, stride, stride, 1]
        bshape = [1, 1, 1, nf]
    elif data_format == 'NCHW':
        channel_ax = 1
        strides = [1, 1, stride, stride]
        bshape = [1, nf, 1, 1]
    else:
        raise NotImplementedError
    nin = x.get_shape()[channel_ax].value
    wshape = [rf, rf, nin, nf]
    with tf.variable_scope(scope):
        w = tf.get_variable("w", wshape, initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [1, nf, 1, 1], initializer=tf.constant_initializer(0.0))
        if data_format == 'NHWC': b = tf.reshape(b, bshape)
        return b + tf.nn.conv2d(x, w, strides=strides, padding=pad, data_format=data_format)
def conv_to_fc(x):
    nh = np.prod([v.value for v in x.get_shape()[1:]])
    x = tf.reshape(x, [-1, nh])
    return x    
def fc(x, scope, nh, *, init_scale=1.0, init_bias=0.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias))
        return tf.matmul(x, w)+b
def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init
