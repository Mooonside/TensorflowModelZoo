import tensorflow as tf

from datasets.voc.pascal_voc_utils import pascal_voc_palette
from tf_ops.wrap_ops import tensor_shape


####################################
# Segmentation Visualization Tools #
####################################


def paint(predictions, palette=pascal_voc_palette):
    num_classes = palette.shape[0]
    paint_ = tf.one_hot(predictions, depth=num_classes, axis=-1, dtype=predictions.dtype)
    paint_ = tf.squeeze(tf.tensordot(paint_, tf.cast(palette, predictions.dtype), axes=[[-1], [0]]), axis=3)
    return paint_


def compare(predictions, labels):
    if tensor_shape(predictions) != tensor_shape(labels):
        h, w = tensor_shape(labels)[1:3]
        predictions = tf.image.resize_nearest_neighbor(predictions, [h, w], align_corners=True)

    same = tf.logical_or(
        tf.equal(predictions, labels),
        tf.equal(labels, 255)
    )
    same = tf.cast(same, tf.int32)
    paint_ = tf.one_hot(same, depth=2, axis=-1, dtype=predictions.dtype)
    paint_ = tf.squeeze(tf.tensordot(paint_, tf.cast([[255, 0, 0], [0, 0, 0]], predictions.dtype), axes=[[-1], [0]]),
                        axis=3)
    return paint_


def locate_boundary(labels):
    """ locate boundaries in labels
    # TODO: Validate this function
    :param labels: [N, H, W, C]
    :return: a bool tensor, true indicating boundaries
    """
    h, w = tensor_shape(labels)[1:3]
    pad = tf.pad(labels, [[0, 0], [0, 1], [0, 0], [0, 0]], mode='REFLECT')[:, 1:, :, :]
    boundary = tf.equal(pad, labels)
    pad = tf.pad(labels, [[0, 0], [0, 0], [0, 1], [0, 0]], mode='REFLECT')[:, :, 1:, :]
    boundary = tf.logical_or(boundary, tf.equal(pad, labels))

    expansions = tf.cast(
        tf.pad(labels, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT'),
        tf.bool
    )
    for xmove in [-1, 0, 1]:
        for ymove in [-1, 0, 1]:
            boundary = tf.logical_or(boundary, expansions[:, 1 + xmove:1 + xmove + h, 1 + ymove:1 + ymove + w, :])
    return boundary

#################################
# Detection Visualization Tools #
#################################


def draw_bbox(image, scores, bboxes):
    for class_id in scores.keys():
        if class_id == 0:
            continue

        # class_scores = scores[class_id]
        # [#bboxes, 1, 4] => [#bboxes, 4]
        class_bboxes = tf.squeeze(bboxes[class_id], axis=1)
        # [#bboxes, 4] => [None, #bboxes, 4]
        class_bboxes = tf.expand_dims(class_bboxes, axis=0)
        image = tf.image.draw_bounding_boxes(image, class_bboxes)
    return image
