


import tensorflow as tf
import numpy as np

_FLOATX = tf.float32


def is_sparse(tensor):
    return isinstance(tensor, tf.SparseTensor)


def to_dense(tensor):
    if is_sparse(tensor):
        return tf.sparse_tensor_to_dense(tensor)
    else:
        return tensor


def ndim(x):
    '''Returns the number of axes in a tensor, as an integer.
    '''
    if is_sparse(x):
        return int(x.shape.get_shape()[0])

    dims = x.get_shape()._dims
    if dims is not None:
        return len(dims)
    return None


def concatenate(tensors, axis=-1):
    '''Concantes a list of tensors alongside the specified axis.
    '''
    if axis < 0:
        dims = ndim(tensors[0])
        if dims:
            axis = axis % dims
        else:
            axis = 0

    if all([is_sparse(x) for x in tensors]):
        return tf.sparse_concat(axis, tensors)
    else:
        return tf.concat(axis, [to_dense(x) for x in tensors])


def random_normal(shape, mean=0.0, std=1.0, dtype=_FLOATX, seed=None):
    if seed is None:
        seed = np.random.randint(10e6)
    return tf.random_normal(shape, mean=mean, stddev=std,
                            dtype=dtype, seed=seed)


def random_uniform(shape, low=0.0, high=1.0, dtype=_FLOATX, seed=None):
    if seed is None:
        seed = np.random.randint(10e6)
    return tf.random_uniform(shape, minval=low, maxval=high,
                             dtype=dtype, seed=seed)


def random_binomial(shape, p=0.0, dtype=_FLOATX, seed=None):
    if seed is None:
        seed = np.random.randint(10e6)
    return tf.select(tf.random_uniform(shape, dtype=dtype, seed=seed) <= p,
                     tf.ones(shape), tf.zeros(shape))


def random_uniform_variable(shape, low=-0.05, high=0.05, dtype=_FLOATX,
                            name=None, seed=None):
    shape = tuple(map(int, shape))
    if seed is None:
        # ensure that randomness is conditioned by the Numpy RNG
        seed = np.random.randint(10e8)
    value = tf.random_uniform_initializer(
        low, high, dtype=dtype, seed=seed)(shape)
    return tf.Variable(value, dtype=dtype, name=name)

