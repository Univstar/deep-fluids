from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak*x)
   
def conv2d(x, o_dim, data_format='NHWC', name=None, k=4, s=2, act=None):
    return slim.conv2d(x, o_dim, k, stride=s, activation_fn=act, scope=name, data_format=data_format)

def linear(x, o_dim, name=None, act=None):
    return slim.fully_connected(x, o_dim, activation_fn=act, scope=name)

def resize_nearest_neighbor(x, new_size, data_format='NHWC'):
    if data_format == 'NCHW':
        x = nchw_to_nhwc(x)
        x = tf.image.resize_nearest_neighbor(x, new_size)
        x = nhwc_to_nchw(x)
    else:
        x = tf.image.resize_nearest_neighbor(x, new_size)
    return x

def upscale(x, scale, data_format='NHWC'):
    _, h, w, _ = get_conv_shape(x, data_format)
    return resize_nearest_neighbor(x, (h*scale, w*scale), data_format)

def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def get_conv_shape(tensor, data_format='NHWC'):
    shape = int_shape(tensor)
    # always return [N, H, W, C]
    if data_format == 'NCHW':
        return [shape[0], shape[2], shape[3], shape[1]]
    elif data_format == 'NHWC':
        return shape

def nchw_to_nhwc(x):
    return tf.transpose(x, [0, 2, 3, 1])

def nhwc_to_nchw(x):
    return tf.transpose(x, [0, 3, 1, 2])

def to_nhwc(image, data_format='NHCW'):
    if data_format == 'NCHW':
        new_image = nchw_to_nhwc(image)
    else:
        new_image = image
    return new_image

def add_channels(x, num_ch=1, data_format='NHWC'):
    b, h, w, c = get_conv_shape(x, data_format)
    if data_format == 'NCHW':
        x = tf.concat([x, tf.zeros([b, num_ch, h, w])], axis=1)
    else:
        x = tf.concat([x, tf.zeros([b, h, w, num_ch])], axis=-1)
    return x

def remove_channels(x, data_format='NHWC'):
    b, h, w, c = get_conv_shape(x, data_format)
    if data_format == 'NCHW':
        x, _ = tf.split(x, [3, -1], axis=1)
    else:
        x, _ = tf.split(x, [3, -1], axis=3)
    return x

def denorm_img(norm, data_format='NHWC'):
    _, _, _, c = get_conv_shape(norm, data_format)
    if c == 2:
        norm = add_channels(norm, num_ch=1, data_format=data_format)
    elif c > 3:
        norm = remove_channels(norm, data_format=data_format)
    img = tf.cast(tf.clip_by_value(to_nhwc((norm + 1)*127.5, data_format), 0, 255), tf.uint8)
    return img

def reshape(x, h, w, c, data_format='NHWC'):
    if data_format == 'NCHW':
        x = tf.reshape(x, [-1, c, h, w])
    else:
        x = tf.reshape(x, [-1, h, w, c])
    return x

def jacobian(x, data_format='NHCW'):
    if data_format == 'NCHW':
        x = nchw_to_nhwc(x)

    dudx = x[:,:,1:,0] - x[:,:,:-1,0]
    dudy = x[:,1:,:,0] - x[:,:-1,:,0]
    dvdx = x[:,:,1:,1] - x[:,:,:-1,1]
    dvdy = x[:,1:,:,1] - x[:,:-1,:,1]
    
    dudx = tf.concat([dudx,tf.expand_dims(dudx[:,:,-1], axis=2)], axis=2)
    dvdx = tf.concat([dvdx,tf.expand_dims(dvdx[:,:,-1], axis=2)], axis=2)
    dudy = tf.concat([dudy,tf.expand_dims(dudy[:,-1,:], axis=1)], axis=1)
    dvdy = tf.concat([dvdy,tf.expand_dims(dvdy[:,-1,:], axis=1)], axis=1)

    j = tf.stack([dudx,dudy,dvdx,dvdy], axis=-1)
    w = tf.expand_dims(dvdx - dudy, axis=-1) # vorticity (for visualization)

    if data_format == 'NCHW':
        j = nhwc_to_nchw(j)
        w = nhwc_to_nchw(w)
    return j, w
    
def curl(x, data_format='NHWC'):
    if data_format == 'NCHW': x = nchw_to_nhwc(x)

    u = x[:,1:,:,0] - x[:,:-1,:,0] # ds/dy
    v = x[:,:,:-1,0] - x[:,:,1:,0] # -ds/dx,
    u = tf.concat([u, tf.expand_dims(u[:,-1,:], axis=1)], axis=1)
    v = tf.concat([v, tf.expand_dims(v[:,:,-1], axis=2)], axis=2)
    c = tf.stack([u,v], axis=-1)

    if data_format == 'NCHW': c = nhwc_to_nchw(c)
    return c

def vort_np(x):
    dvdx = x[:,:,1:,1] - x[:,:,:-1,1]
    dudy = x[:,1:,:,0] - x[:,:-1,:,0]
    dvdx = np.concatenate([dvdx,np.expand_dims(dvdx[:,:,-1], axis=2)], axis=2)
    dudy = np.concatenate([dudy,np.expand_dims(dudy[:,-1,:], axis=1)], axis=1)
    return np.expand_dims(dvdx - dudy, axis=-1)

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
