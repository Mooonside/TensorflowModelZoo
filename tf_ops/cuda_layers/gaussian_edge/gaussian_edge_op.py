from __future__ import absolute_import
import tensorflow as tf
import os.path as osp
from tensorflow.python.framework import ops
from tf_ops.wrap_ops import tensor_rank, tensor_shape

filename = osp.join(osp.dirname(__file__), 'gaussian_edge.so')
gaussian_edge_module = tf.load_op_library(filename)
gaussian_edge_op = gaussian_edge_module.gaussian_edge_op


def gaussian_edge(
        input,
        kernel=(3, 3),
        sigma=None,
        nearest=3,
        dtype=tf.float32
):
    """
    For each point in input, if input is larger than 0, assign a gaussian distribution around the point
    For multiple gaussian, take their maximum!

    :param input: [N, H, W] or [N, H, W, 1]
    :param kernel: [kh, kw]
    :param sigma: [sh, sw]
    :param dtype: output type
    :return:
    """
    assert input.dtype is tf.int32
    add_tail_axis = False
    if tensor_rank(input) == 4:
        if tensor_shape(input)[-1] == 1:
            input = input[..., 0]
            add_tail_axis = True

    #  sigma = 0.3\*((ksize-1)\*0.5 - 1) + 0.8
    if sigma is None:
        sigmax = 0.3 * (kernel[0] * 0.5 - 1) + 0.8
        sigmay = 0.3 * (kernel[1] * 0.5 - 1) + 0.8
        sigma = [sigmax, sigmay]

    edge = gaussian_edge_op(x=input,
                            T=dtype,
                            kernel=kernel,
                            sigma=sigma,
                            nearest=nearest)
    if add_tail_axis:
        edge = edge[..., tf.newaxis]

    return edge


if __name__ == '__main__':
    import numpy as np
    constant = np.zeros([2, 5, 5], dtype=np.int32)
    constant[0, 2, 3] = 1
    constant[0, 2, 4] = 1

    data = tf.get_variable(
        name='data',
        initializer=constant,
    )
    with tf.device('/CPU:0'):
        out = gaussian_edge(
            input=data,
            kernel=[3, 3],
            sigma=[0.5, 0.5],
            dtype=tf.float32,
        )

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    y = sess.run(out)

    print(y)
