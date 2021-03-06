from __future__ import absolute_import
import tensorflow as tf
import os.path as osp
from tensorflow.python.framework import ops


filename = osp.join(osp.dirname(__file__), 'deform_conv.so')
_deform_conv_module = tf.load_op_library(filename)
deform_conv_op = _deform_conv_module.deform_conv_op
deform_conv_grad_op = _deform_conv_module.deform_conv_backprop_op


@ops.RegisterGradient("DeformConvOp")
def _deform_conv_grad(op, grad):
  """The gradients for `deform_conv`.
  Args:
    op: The `deform_conv` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `roi_pool` op.
  Returns:
    Gradients with respect to the input of `zero_out`.
  """
  data = op.inputs[0]
  filter = op.inputs[1]
  offset = op.inputs[2]
  
  strides = op.get_attr('strides')
  rates = op.get_attr('rates')
  num_groups = op.get_attr('num_groups')
  padding = op.get_attr('padding')
  data_format = op.get_attr('data_format')
  deformable_group = op.get_attr('deformable_group')  

  # compute gradient
  data_grad = deform_conv_grad_op(data, filter, offset, grad, strides, rates, num_groups, deformable_group, padding, data_format)

  return data_grad  # List of one Tensor, since we have one input


if __name__ == '__main__':
    import numpy as np
    inputs = np.arange(5 * 5).reshape([1, 1, 5, 5]).astype(np.float32)
    filters = np.ones([1, 1, 3, 3]).astype(np.float32)
    offsets = np.zeros([1, 3*3*2, 5, 5]).astype(np.float32)

    d = deform_conv_op(x=inputs,
                   filter=filters,
                   offset=offsets,
                   strides=[1, 1, 1, 1],
                   rates=[1, 1, 2, 2],
                   padding='VALID',
                   num_groups=1,
                   deformable_group=1,
                   data_format='NCHW',
                   name='deform_conv')

    sess = tf.Session()
    # inputs_v, d_v = sess.run([inputs, d])
    d_v = sess.run(d)
    print(d_v.shape, d)
    print(d_v)

    # print(inputs_v)