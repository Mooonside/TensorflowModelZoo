from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
from functools import partial as set_parameter

import tensorflow as tf
from numpy import arange
from numpy.random import shuffle
from tensorflow.python.ops import image_ops

from tf_ops.augmentation import distort_color, flip_image_left_right, apply_random_op

# Basic model parameters as external flags.
FLAGS = None

# Constants used for dealing with the files, matches convert_to_records.
# TRAIN_DIR = '/data/chenyifeng/TF_Records/ILSVRC_2015_CLS/train'
# VALIDATION_DIR = '/data/chenyifeng/TF_Records/ILSVRC_2015_CLS/val'

TRAIN_DIR = '/mnt/disk50_CHENYIFENG/TF_Records/ILSVRC_2015_CLS/train'
VALIDATION_DIR = '/mnt/disk50_CHENYIFENG/TF_Records/ILSVRC_2015_CLS/val'

TRAIN_NUM = 1281167
VALID_NUM = 50000


def decode(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image/height': tf.FixedLenFeature([1], tf.int64),
            'image/width': tf.FixedLenFeature([1], tf.int64),
            'image/colorspace': tf.FixedLenFeature([], tf.string),
            'image/channels': tf.FixedLenFeature([1], tf.int64),
            'image/class/label': tf.FixedLenFeature([1], tf.int64),
            'image/class/synset': tf.FixedLenFeature([], tf.string),
            'image/class/text': tf.FixedLenFeature([], tf.string),
            'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
            'image/object/bbox/label': tf.VarLenFeature(tf.int64),
            'image/format': tf.FixedLenFeature([], tf.string),
            'image/filename': tf.FixedLenFeature([], tf.string),
            'image/encoded': tf.FixedLenFeature([], tf.string)
        })
    return features


def extract(features):
    image = features['image/encoded']

    image = image_ops.decode_jpeg(image, channels=3)
    label = features['image/class/label']
    ymin = tf.sparse_tensor_to_dense(features['image/object/bbox/ymin'])
    xmin = tf.sparse_tensor_to_dense(features['image/object/bbox/xmin'])
    ymax = tf.sparse_tensor_to_dense(features['image/object/bbox/ymax'])
    xmax = tf.sparse_tensor_to_dense(features['image/object/bbox/xmax'])
    bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
    name = features['image/filename']

    return name, image, label, bboxes


def normalize(name, image, label):
    # Convert from [0, 255] -> [-1.0, 1.0] floats.
    image = (2.0 / 255.0) * image - 1.0
    return name, image, label


def reshape(name, image, label, bboxes, reshape_size=None):
    if reshape_size is not None:
        image = tf.expand_dims(image, axis=0)
        image = tf.image.resize_bilinear(image, reshape_size)
        image = tf.squeeze(image, axis=0)
    image.set_shape(reshape_size + [3])
    # remove bboxes return here
    return name, image, label


def cast_type(name, image, label, bboxes):
    return name, tf.cast(image, tf.float32), tf.cast(label, tf.int32), bboxes


def inception_augmentation(name, image, label, bboxes, visualization=False):
    bbox_begin, bbox_size, distort_bbox = \
        tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=tf.expand_dims(bboxes, 0),
            min_object_covered=0.1,
            aspect_ratio_range=[0.75, 1.33],
            area_range=[0.05, 1.0],
            max_attempts=100,
            use_image_if_no_bounding_boxes=True)
    # Crop the image to the specified bounding box.
    cropped_image = tf.slice(image, bbox_begin, bbox_size)
    if visualization:
        image_with_distorted_box = tf.image.draw_bounding_boxes(
            tf.expand_dims(image, 0), distort_bbox)
        tf.summary.image('images_with_distorted_bounding_box',
                         image_with_distorted_box)

    # Restore the shape since the dynamic slice loses 3rd dimension.
    cropped_image.set_shape([None, None, 3])
    flipped_image = apply_random_op(flip_image_left_right, {'image': cropped_image})
    distort_image = distort_color(flipped_image)
    # no use for this return
    return name, distort_image, label, bboxes


def eval_preprocess(name, image, label, bboxes):
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    image = tf.image.central_crop(image, central_fraction=0.875)
    return name, image, label, bboxes


def get_dataset(dir, batch_size, num_epochs, reshape_size,
                num_readers=1, augment_func=inception_augmentation):
    if not num_epochs:
        num_epochs = None
    filenames = [os.path.join(dir, i) for i in os.listdir(dir)]
    shuffle_idx = arange(len(filenames))
    shuffle(shuffle_idx)

    filenames = [filenames[shuffle_idx[idx]] for idx in range(len(filenames))]

    with tf.name_scope('input'):
        # TFRecordDataset opens a protobuf and reads entries line by line
        # could also be [list, of, filenames]
        dataset = tf.data.TFRecordDataset(filenames,
                                          num_parallel_reads=num_readers)
        # dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.repeat(num_epochs)

        # map takes a python function and applies it to every sample
        dataset = dataset.map(decode)
        dataset = dataset.map(extract)
        dataset = dataset.map(cast_type)

        if augment_func is not None:
            print('Enabling Augmentation...')
            dataset = dataset.map(augment_func)
        dataset = dataset.map(set_parameter(reshape, reshape_size=reshape_size))

        dataset = dataset.map(normalize)

        # the parameter is the queue size
        dataset = dataset.shuffle(1000 + 3 * batch_size)
        dataset = dataset.batch(batch_size)
    return dataset


def get_next_batch(dataset):
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

# if __name__ == '__main__':
#     with tf.device('/CPU:0'):
#         dataset = get_dataset(dir=TRAIN_DIR,
#                               batch_size=16,
#                               num_epochs=1,
#                               reshape_size=[513, 513])
#         name_batch, image_batch, label_batch = get_next_batch(dataset)
#         sess = tf.Session()
#         name_batch_v, image_batch_v, label_batch_v = sess.run([name_batch, image_batch, label_batch])
#         print(image_batch_v)
