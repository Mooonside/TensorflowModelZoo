"""
An implementation of VGG-16
By Yifeng Chen
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_ops.wrap_ops import conv2d, layers_lib, max_pool2d
from tensorflow.contrib.layers.python.layers import utils
import tensorflow as tf


arg_scope = tf.contrib.framework.arg_scope


def vgg_arg_scope(weight_init=None,
                  weight_reg=None,
                  bias_init=tf.zeros_initializer,
                  bias_reg=None,
                  is_training=True):
    """
    define arg_scope for vgg model
    :param weight_init: weight initializer
    :param weight_reg: weight regularizer
    :param bias_init: bias initializer
    :param bias_reg: bias regularizer
    :param is_training: whether training the model
    :return: arg_scope
    """
    with arg_scope([conv2d, layers_lib.fully_connected],
                   activation_fn=tf.nn.relu,
                   weights_initializer=weight_init,
                   weights_regularizer=weight_reg,
                   biases_initializer=bias_init,
                   biases_regularizer=bias_reg):
        with arg_scope([conv2d], padding='SAME'):
            with arg_scope([layers_lib.dropout], is_training=is_training) as arg_sc:
                return arg_sc


def vgg_conv_block(inputs, outc, times, scope, ksize = 3):
    """
    Repeat conv2d with [3, 3] kernel for times
    :param inputs:
    :param outc:
    :param times:
    :param scope:
    :param ksize:
    :return:
    """
    net = inputs
    with tf.variable_scope(scope, 'conv'):
        for i in range(times):
            iname = scope + '_' + str(i + 1)
            net = conv2d(net, outc, ksize, scope=iname)
    return net


def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16',
           fc_conv_padding='VALID',
           global_pool=False):
    """Oxford Net VGG 16-Layers version D Example.

    Note: All the fully_connected layers have been transformed to conv2d layers.
          To use in classification mode, resize input to 224x224.

    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      num_classes: number of predicted classes. If 0 or None, the logits layer is
        omitted and the input features to the logits layer are returned instead.
      is_training: whether or not the model is being trained.
      dropout_keep_prob: the probability that activations are kept in the dropout
        layers during training.
      spatial_squeeze: whether or not should squeeze the spatial dimensions of the
        outputs. Useful to remove unnecessary dimensions for classification.
      scope: Optional scope for the variables.
      fc_conv_padding: the type of padding to use for the fully connected layer
        that is implemented as a convolutional layer. Use 'SAME' padding if you
        are applying the network in a fully convolutional manner and want to
        get a prediction map downsampled by a factor of 32 as an output.
        Otherwise, the output prediction map will be (input / 32) - 6 in case of
        'VALID' padding.
      global_pool: Optional boolean flag. If True, the input to the classification
        layer is avgpooled to size 1x1, for any input size. (This is not part
        of the original VGG architecture.)

    Returns:
      net: the output of the logits layer (if num_classes is a non-zero integer),
        or the input to the logits layer (if num_classes is 0 or None).
      end_points: a dict of tensors with intermediate activations.
    """
    with tf.variable_scope(scope, 'vgg_16', [inputs], reuse=tf.AUTO_REUSE) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with arg_scope([conv2d, layers_lib.fully_connected, ],
                       outputs_collections=end_points_collection):
            net = vgg_conv_block(inputs=inputs, outc=64, times=2, scope='conv1')
            net = max_pool2d(net, 2, scope='pool1')
            net = vgg_conv_block(inputs=net, outc=128, times=2, scope='conv2')
            net = max_pool2d(net, 2, scope='pool2')
            net = vgg_conv_block(inputs=net, outc=256, times=3, scope='conv3')
            net = max_pool2d(net, 2, scope='pool3')
            net = vgg_conv_block(inputs=net, outc=512, times=3, scope='conv4')
            net = max_pool2d(net, 2, scope='pool4')
            net = vgg_conv_block(inputs=net, outc=512, times=3, scope='conv5')
            net = max_pool2d(net, 2, scope='pool5')
            #
            # Use conv2d instead of fully_connected layers.
            net = conv2d(net, 4096, 7, padding=fc_conv_padding, scope='fc6')
            net = layers_lib.dropout(net, keep_prob=dropout_keep_prob, is_training=is_training, scope='dropout6')
            net = conv2d(net, 4096, 1, scope='fc7')
            # Convert end_points_collection into a end_point dict.
            end_points = utils.convert_collection_to_dict(end_points_collection)

            if global_pool:
                net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                end_points['global_pool'] = net
            if num_classes:
                net = layers_lib.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')
                net = conv2d(net, num_classes, 1, activation_fn=None, scope='fc8')

                if spatial_squeeze:
                    net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                end_points[sc.name + '/fc8'] = net
            return net, end_points


if __name__ == '__main__':
    sess = tf.Session()
    inputs = tf.placeholder(name='inputs', shape=[16, 224, 224, 3], dtype=tf.float32)
    with arg_scope(vgg_arg_scope()):
        _, end_points = vgg_16(inputs)
    for i in end_points.keys():
        print(i, end_points[i].shape)
    # sess.run(tf.global_variables_initializer())
    # saver = tf.train.Saver()
    # saver.restore(sess=sess, save_path='/home/yifeng/Models/pretrain/vgg_16.ckpt')
    #
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for i in trainable_vars:
        print(i.name, i.shape)
