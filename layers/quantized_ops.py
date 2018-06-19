# -*- coding: utf-8 -*-
from __future__ import absolute_import
import keras.backend as K
import tensorflow as tf
import numpy as np

def round_through(x):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    '''
    rounded = K.round(x)
    return x + K.stop_gradient(rounded - x)


def _hard_sigmoid(x):
    '''Hard sigmoid different from the more conventional form (see definition of K.hard_sigmoid).

    # Reference:
    - [BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    x = (0.5 * x) + 0.5
    return K.clip(x, 0, 1)


def binary_sigmoid(x):
    '''Binary hard sigmoid for training binarized neural network.

    # Reference:
    - [BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    return round_through(_hard_sigmoid(x))


def binary_tanh(x):
    '''Binary hard sigmoid for training binarized neural network.
     The neurons' activations binarization function
     It behaves like the sign function during forward propagation
     And like:
        hard_tanh(x) = 2 * _hard_sigmoid(x) - 1 
        clear gradient when |x| > 1 during back propagation

    # Reference:
    - [BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    return 2 * round_through(_hard_sigmoid(x)) - 1


def binarize(W, H=1):
    '''The weights' binarization function, 

    # Reference:
    - [BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    # [-H, H] -> -H or H
    Wb = H * binary_tanh(W / H)
    return Wb


def _mean_abs(x, axis=None, keepdims=False):
    return K.stop_gradient(K.mean(K.abs(x), axis=axis, keepdims=keepdims))

    
def xnorize(W, H=1., axis=None, keepdims=False):
    Wb = binarize(W, H)
    Wa = _mean_abs(W, axis, keepdims)
    
    return Wa, Wb
    


def clip_through(x, min_val, max_val):
    '''Element-wise clipping with gradient propagation
    Analogue to round_through
    '''
    clipped = K.clip(x, min_val, max_val)
    clipped_through= x + K.stop_gradient(clipped-x)
    return clipped_through 


def clip_through(x, min, max):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    '''
    clipped = K.clip(x,min,max)
    return x + K.stop_gradient(clipped - x)





def quantize(W, nb = 16, clip_through=False):

    '''The weights' binarization function, 

    # Reference:
    - [QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''

    non_sign_bits = nb-1
    m = pow(2,non_sign_bits)
    #W = tf.Print(W,[W],summarize=20)
    if clip_through:
        Wq = clip_through(round_through(W*m),-m,m-1)/m
    else:
        Wq = K.clip(round_through(W*m),-m,m-1)/m
    #Wq = tf.Print(Wq,[Wq],summarize=20)
    return Wq


def quantized_relu(W, nb=16):

    '''The weights' binarization function, 

    # Reference:
    - [QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    #non_sign_bits = nb-1
    #m = pow(2,non_sign_bits)
    #Wq = K.clip(round_through(W*m),0,m-1)/m

    nb_bits = nb
    Wq = K.clip(2. * (round_through(_hard_sigmoid(W) * pow(2, nb_bits)) / pow(2, nb_bits)) - 1., 0,
                1 - 1.0 / pow(2, nb_bits - 1))
    return Wq


def quantized_tanh(W, nb=16):

    '''The weights' binarization function,

    # Reference:
    - [QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    non_sign_bits = nb-1
    m = pow(2,non_sign_bits)
    #W = tf.Print(W,[W],summarize=20)
    Wq = K.clip(round_through(W*m),-m,m-1)/m
    #Wq = tf.Print(Wq,[Wq],summarize=20)
    return Wq

def quantized_leakyrelu(W, nb=16, alpha=0.1):

    '''The weights' binarization function, 

    # Reference:
    - [QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    if alpha != 0.:
        negative_part = tf.nn.relu(-W)
    W = tf.nn.relu(W)
    if alpha != 0.:
        alpha = tf.cast(tf.convert_to_tensor(alpha), W.dtype.base_dtype)
        W -= alpha * negative_part

    non_sign_bits = nb-1
    m = pow(2,non_sign_bits)
    #W = tf.Print(W,[W],summarize=20)
    Wq = clip_through(round_through(W*m),-m,m-1)/m
    #Wq = tf.Print(Wq,[Wq],summarize=20)

    return Wq

def quantized_maxrelu(W, nb=16):

    '''The weights' binarization function, 

    # Reference:
    - [QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    non_sign_bits = nb-1
    max_ = tf.reduce_max((W))
    #max_ = tf.Print(max_,[max_])
    max__ = tf.pow(2.0,tf.ceil(tf.log(max_)/tf.log(tf.cast(tf.convert_to_tensor(2.0), W.dtype.base_dtype))))
    #max__ = tf.Print(max__,[max__])
    m = pow(2,non_sign_bits)
    #W = tf.Print(W,[W],summarize=20)
    Wq = max__*clip_through(round_through(W/max__*(m)),0,m-1)/(m)
    #Wq = tf.Print(Wq,[Wq],summarize=20)

    return Wq

def quantized_leakymaxrelu(W, nb=16, alpha=0.1):

    '''The weights' binarization function, 

    # Reference:
    - [QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    if alpha != 0.:
        negative_part = tf.nn.relu(-W)
    W = tf.nn.relu(W)
    if alpha != 0.:
        alpha = tf.cast(tf.convert_to_tensor(alpha), W.dtype.base_dtype)
        W -= alpha * negative_part

    max_ = tf.reduce_max((W))
    #max_ = tf.Print(max_,[max_])
    max__ = tf.pow(2.0,tf.ceil(tf.log(max_)/tf.log(tf.cast(tf.convert_to_tensor(2.0), W.dtype.base_dtype))))
    #max__ = tf.Print(max__,[max__])

    non_sign_bits = nb-1
    m = pow(2,non_sign_bits)
    #W = tf.Print(W,[W],summarize=20)
    Wq = max__* clip_through(round_through(W/max__*m),-m,m-1)/m
    #Wq = tf.Print(Wq,[Wq],summarize=20)

    return Wq

    
def xnorize_qnn(W, H=1., axis=None, keepdims=False):
    Wb = quantize(W, H)
    Wa = _mean_abs(W, axis, keepdims)
    
    return Wa, Wb

def switch(condition, t, e):
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        return tf.where(condition, t, e)
    elif K.backend() == 'theano':
        import theano.tensor as tt
        return tt.switch(condition, t, e)


def _ternarize(W, H=1):
    '''The weights' ternarization function, 

    # References:
    - [Recurrent Neural Networks with Limited Numerical Precision](http://arxiv.org/abs/1608.06902)
    - [Ternary Weight Networks](http://arxiv.org/abs/1605.04711)
    '''
    W /= H

    ones = K.ones_like(W)
    zeros = K.zeros_like(W)
    Wt = switch(W > 0.5, ones, switch(W <= -0.5, -ones, zeros))

    Wt *= H

    return Wt


def ternarize(W, H=1):
    '''The weights' ternarization function, 

    # References:
    - [Recurrent Neural Networks with Limited Numerical Precision](http://arxiv.org/abs/1608.06902)
    - [Ternary Weight Networks](http://arxiv.org/abs/1605.04711)
    '''
    Wt = _ternarize(W, H)
    return W + K.stop_gradient(Wt - W)


def ternarize_dot(x, W):
    '''For RNN (maybe Dense or Conv too). 
    Refer to 'Recurrent Neural Networks with Limited Numerical Precision' Section 3.1
    '''
    Wt = _ternarize(W)
    return K.dot(x, W) + K.stop_gradient(K.dot(x, Wt - W))
    