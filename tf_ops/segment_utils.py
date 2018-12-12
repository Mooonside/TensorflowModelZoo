from tf_ops.wrap_ops import tensor_shape, tensor_rank
import tensorflow as tf


def from_sem_to_boundary(anno, nrange=3):
    if tensor_rank(anno) == 3:
        anno = anno[tf.newaxis, ...]
    H, W = tensor_shape(anno)[1:3]

    # def generate_boundaries(anno):
    anno = tf.cast(anno, tf.int32)
    is_bound = tf.zeros_like(anno, dtype=tf.bool)

    for r in range(nrange):
        pad_anno = tf.pad(anno, [[0, 0], [r, r], [r, r], [0, 0]], mode="SYMMETRIC")

        shifts = []
        for ridx in [-1 * r, 0, 1 * r]:
            for cidx in [-1 * r, 0, 1 * r]:
                trans_anno = pad_anno[:, (r + ridx):(H + r + ridx),
                             (r + cidx):(W + r + cidx):]
                shifts.append(trans_anno)

        for shift in shifts:
            shift_boundary = tf.not_equal(shift, anno)
            is_bound = tf.logical_or(is_bound, shift_boundary)

    return is_bound