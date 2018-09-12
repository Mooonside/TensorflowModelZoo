import numpy as np
import tensorflow as tf

slim = tf.contrib.slim
from tensorflow.contrib.layers.python.layers import conv2d
from tf_ops.wrap_ops import tensor_shape, smooth_l1

POS_NUMBER_PER_LAYER_SCOPE = 'POS_NUMBER_PER_LAYER_SCOPE'
NEG_NUMBER_PER_LAYER_SCOPE = 'NEG_NUMBER_PER_LAYER_SCOPE'
ZERO_NUMBER_PER_LAYER_SCOPE = 'ZERO_NUMBER_PER_LAYER_SCOPE'



def bboxs_clip(bboxes, vmin=0, vmax=1):
    bboxes = tf.maximum(bboxes, np.asarray([vmin, vmin, vmin, vmin]).reshape([-1, 4]))
    bboxes = tf.minimum(bboxes, np.asarray([vmax, vmax, vmax, vmax]).reshape([-1, 4]))
    return bboxes


def iou(bboxes, ref_box):
    """
    :param bboxes: [N, 4]
    :param ref_box: [1, 4]
    :return:
    """
    bboxes_ymin, bboxes_xmin, bboxes_ymax, bboxes_xmax = tf.unstack(bboxes, axis=-1)

    ref_ymin = ref_box[0]
    ref_xmin = ref_box[1]
    ref_ymax = ref_box[2]
    ref_xmax = ref_box[3]

    bboxes_volume = tf.maximum(bboxes_xmax - bboxes_xmin, 0) * \
                    tf.maximum(bboxes_ymax - bboxes_ymin, 0)

    ref_volume = tf.maximum(ref_xmax - ref_xmin, 0) * \
                 tf.maximum(ref_ymax - ref_ymin, 0)

    int_ymin = tf.maximum(bboxes_ymin, ref_ymin)
    int_xmin = tf.maximum(bboxes_xmin, ref_xmin)

    int_ymax = tf.minimum(bboxes_ymax, ref_ymax)
    int_xmax = tf.minimum(bboxes_xmax, ref_xmax)

    int_volume = tf.maximum(int_xmax - int_xmin, 0) * \
                 tf.maximum(int_ymax - int_ymin, 0)

    ious = int_volume / (bboxes_volume + ref_volume - int_volume + 1e-20)
    return ious


def bboxes_intersection(bboxes, bbox_ref, name=None):
    """Compute relative intersection between a reference box and a
    collection of bounding boxes. Namely, compute the quotient between
    intersection area and box area.

    Args:
      bbox_ref: (N, 4) or (4,) Tensor with reference bounding box(es).
      bboxes: (N, 4) Tensor, collection of bounding boxes.
    Return:
      (N,) Tensor with relative intersection.
    """
    with tf.name_scope(name, 'bboxes_intersection'):
        # Should be more efficient to first transpose.
        bboxes = tf.transpose(bboxes)
        bbox_ref = tf.transpose(bbox_ref)
        # Intersection bbox and volume.
        int_ymin = tf.maximum(bboxes[0], bbox_ref[0])
        int_xmin = tf.maximum(bboxes[1], bbox_ref[1])
        int_ymax = tf.minimum(bboxes[2], bbox_ref[2])
        int_xmax = tf.minimum(bboxes[3], bbox_ref[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        inter_vol = h * w
        bboxes_vol = (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1])
        scores = safe_divide(inter_vol, bboxes_vol, 'intersection')
        tf.add_to_collection('debug_scores', scores)
        return scores


def bboxes_resize(bboxes, bbox_ref, name=None):
    with tf.name_scope(name, 'bboxes_resize'):
        # Translate.
        v = tf.stack([bbox_ref[0], bbox_ref[1], bbox_ref[0], bbox_ref[1]])
        bboxes = bboxes - v

        # Scale.
        s = tf.stack([bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1],
                      bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1]])
        bboxes = bboxes / s
        return bboxes


def soft_nms(scores, bboxes, max_output_size, sigma=0.5):
    def gaussian_decay(score, degree, sigma=1.0):
        return score * tf.exp(- degree ** 2 / sigma)

    bboxes_num = tensor_shape(scores)[0]
    loop_times = min(bboxes_num, max_output_size)
    is_select = tf.zeros(shape=[bboxes_num], dtype=tf.float32)

    def condition(i, scores, is_select):
        return tf.less(i, loop_times)

    def main_body(i, scores, is_select):
        idx = tf.argmax(scores * (1 - is_select))
        # mark idx as one
        is_select = is_select + \
                    tf.cast(tf.one_hot(idx, bboxes_num), tf.float32)

        ious = iou(bboxes, tf.gather(bboxes, idx))
        decay_scores = gaussian_decay(scores, ious, sigma=sigma)

        scores = is_select * scores + (1 - is_select) * decay_scores
        return [i + 1, scores, is_select]

    i = 0
    [i, scores, is_select] = tf.while_loop(
        cond=condition,
        body=main_body,
        loop_vars=[i, scores, is_select])
    # [?,]
    idxes = tf.squeeze(tf.where(is_select > 0), axis=-1)
    sorted_scores, sorted_idx = tf.nn.top_k(tf.gather(scores, idxes), k=loop_times)
    sorted_bbox = tf.gather(bboxes, tf.gather(idxes, sorted_idx))

    return sorted_scores, sorted_bbox


def _layer_anchors(feature_shape, anchors, offset=0.5):
    """
    For feature shape, encode them into scale bboxes with different ratios
    :param feature_shape: [H ,W]
    :param scale_c: current layer bbox scale
    :param scale_n: next layer bbox scale
    :param ratios: different aspect ratios
    :return: y, x, h ,w
    """
    # x in [ (0...fw), ..., (0....fw)]
    # y in [ (0...0), ..., (fh...fh)]
    y, x = np.mgrid[0:feature_shape[0], 0:feature_shape[1]]
    # support broadcasting in encoding part
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)
    # may have (kernel - 1) / 2 pixels misalignment
    y = (y + offset) / feature_shape[0]
    x = (x + offset) / feature_shape[1]
    anchors = np.asarray(anchors)
    h = anchors[:, 0]
    w = anchors[:, 1]

    return y, x, np.asarray(h, np.float32), np.asarray(w, np.float32)


def _layer_prediction(feature_map, num_anchors, conv_params, num_classes, scope=None):
    """
    For each location in feature map, predict 4*num_anchors locations and num_classes objectness
    :param feature_map: [None, H, W, C]
    :param scope:
    :return: locations with shape [None, H, W, num_anchors, 4]
             scores with shape [None, H, W, num_anchors, num_classes]
    """
    with tf.variable_scope(scope, 'feature2bbox'):
        # TODO : CHECK ACTIVATION FUNC HERE

        with slim.arg_scope([conv2d],
                            activation_fn=None,
                            normalizer_fn=None,
                            **conv_params):
            locations = conv2d(feature_map,
                               kernel_size=3,
                               num_outputs=num_anchors * 4,
                               scope='conv_loc')

            scores = conv2d(feature_map,
                            kernel_size=3,
                            num_outputs=num_anchors * num_classes,
                            scope='conv_obj')

        partial_shape = (tensor_shape(locations))[1:-1]

        locations = tf.reshape(locations, shape=[-1] + partial_shape + [num_anchors, 4])
        scores = tf.reshape(scores, shape=[-1] + partial_shape + [num_anchors, num_classes])

        return locations, scores


def _layer_encoding(layer_anchors, labels, bboxes, background_label=0, prior_scale=None, central_responsible=False,
                    only_max=False):
    anchors_cy, anchors_cx, anchors_h, anchors_w = layer_anchors
    # support broadcasting
    anchors_ymin = anchors_cy - (anchors_h / 2.0)
    anchors_xmin = anchors_cx - (anchors_w / 2.0)
    anchors_ymax = anchors_cy + (anchors_h / 2.0)
    anchors_xmax = anchors_cx + (anchors_w / 2.0)

    # convert into four corners
    anchors_volume = anchors_h * anchors_w

    assert anchors_cy.shape == anchors_cx.shape and anchors_h.shape == anchors_w.shape

    # [fh, fW, num_anchors]
    anchors_shape = anchors_ymin.shape

    # for each anchor, assign a label for it
    # Steps:
    #   1. for each gt bbox, solve for jaccard idx of anchors with it
    #   2. assign each anchor
    encode_labels = background_label * tf.ones(anchors_shape, dtype=tf.int32)
    encode_ious = tf.zeros(anchors_shape, dtype=tf.float32)
    encode_ymin = tf.zeros(anchors_shape, dtype=tf.float32)
    encode_xmin = tf.zeros(anchors_shape, dtype=tf.float32)
    encode_ymax = tf.ones(anchors_shape, dtype=tf.float32)
    encode_xmax = tf.ones(anchors_shape, dtype=tf.float32)

    def condition(idx, bboxes, encode_labels, encode_ious, encode_ymin, encode_xmin, encode_ymax, encode_xmax):
        return tf.less(idx, tf.shape(bboxes)[0])

    def body(idx, bboxes, encode_labels, encode_ious, encode_ymin, encode_xmin, encode_ymax, encode_xmax):
        # keep retrieve order the same as reading order in dataset!
        bbox_ymin, bbox_xmin, bbox_ymax, bbox_xmax = tf.unstack(bboxes[idx, :])
        bbox_volume = (bbox_ymax - bbox_ymin) * (bbox_xmax - bbox_xmin)

        inter_ymin = tf.maximum(bbox_ymin, anchors_ymin)
        inter_xmin = tf.maximum(bbox_xmin, anchors_xmin)
        inter_ymax = tf.minimum(bbox_ymax, anchors_ymax)
        inter_xmax = tf.minimum(bbox_xmax, anchors_xmax)

        inter_volume = tf.maximum(inter_ymax - inter_ymin, 0) * \
                       tf.maximum(inter_xmax - inter_xmin, 0)

        selector = tf.cast(tf.not_equal(inter_volume, 0), tf.float32)

        denominator = selector * (anchors_volume + bbox_volume - inter_volume) + (1 - selector)

        ious = inter_volume / (denominator)

        # update
        selector = tf.cast(ious > encode_ious, tf.int32)
        # set central responsible mask
        if central_responsible:
            bbox_cy = tf.cast(tf.floor((bbox_ymax + bbox_ymin) * anchors_shape[0] / 2), tf.int32)
            bbox_cx = tf.cast(tf.floor((bbox_xmax + bbox_xmin) * anchors_shape[1] / 2), tf.int32)

            coords_x, coords_y = tf.meshgrid(tf.range(0, anchors_shape[1], dtype=tf.int32),
                                             tf.range(0, anchors_shape[0], dtype=tf.int32))

            mask = tf.cast(tf.logical_and(
                tf.equal(coords_x, bbox_cx),
                tf.equal(coords_y, bbox_cy)
            ), tf.int32)
            selector *= tf.expand_dims(mask, axis=-1)

        encode_labels = selector * labels[idx] + (1 - selector) * encode_labels
        selector = tf.cast(selector, tf.float32)
        encode_ymin = selector * bbox_ymin + (1 - selector) * encode_ymin
        encode_xmin = selector * bbox_xmin + (1 - selector) * encode_xmin
        encode_ymax = selector * bbox_ymax + (1 - selector) * encode_ymax
        encode_xmax = selector * bbox_xmax + (1 - selector) * encode_xmax

        encode_ious = tf.maximum(encode_ious, ious)

        return [idx + 1, bboxes, encode_labels, encode_ious, encode_ymin, encode_xmin, encode_ymax, encode_xmax]

    idx = 0
    [idx, bboxes, encode_labels, encode_ious, encode_ymin, encode_xmin, encode_ymax, encode_xmax] = \
        tf.while_loop(cond=condition, body=body, loop_vars=[idx, bboxes,
                                                            encode_labels,
                                                            encode_ious,
                                                            encode_ymin,
                                                            encode_xmin,
                                                            encode_ymax,
                                                            encode_xmax])

    # reform to center, size pattern
    encode_cy = (encode_ymin + encode_ymax) / 2
    encode_cx = (encode_xmin + encode_xmax) / 2
    encode_h = encode_ymax - encode_ymin
    encode_w = encode_xmax - encode_xmin
    # Do Bbox regression here
    # [h , w, c]  = ([h , w, c] - [h , w, 1]) / [c]

    if central_responsible:
        # encoding for t_y = sigmoid(p_y) + c_y
        encode_cy = (encode_cy - anchors_cy)
        encode_cx = (encode_cx - anchors_cx)
    else:
        # encoding for t_y = c_h * p_y + c_y
        encode_cy = (encode_cy - anchors_cy) / anchors_h
        encode_cx = (encode_cx - anchors_cx) / anchors_w

    # encoding for t_h = a_h * exp(p_h)
    encode_h = tf.log(encode_h / anchors_h)
    encode_w = tf.log(encode_w / anchors_w)

    if prior_scale is not None:
        encode_cy *= prior_scale[0]
        encode_cx *= prior_scale[1]
        encode_h *= prior_scale[2]
        encode_w *= prior_scale[3]

    # 4 channels are in order y, x, h, w
    encode_locations = tf.stack([encode_cy, encode_cx, encode_h, encode_w], axis=-1)
    return encode_locations, encode_labels, encode_ious


def _layer_decode(locations, layer_anchors, clip=True, prior_scale=None, central_responsible=False):
    """
    Do bbox regression according to anchors positions and scales
    :param locations: [None, H, W, ,K ,4]
    :param layer_anchors: y[H, W, 1], x[H, W, 1], h[K], w[K]
    :param clip: whether clip the decode boxes in image
    :return: [H, W, K, 4]
    """
    anchors_y, anchors_x, anchors_h, anchors_w = layer_anchors
    # [None, H, W, K]
    pred_y, pred_x, pred_h, pred_w = tf.unstack(locations, axis=-1)
    if prior_scale is not None:
        pred_y /= prior_scale[0]
        pred_x /= prior_scale[1]
        pred_h /= prior_scale[2]
        pred_w /= prior_scale[3]
    # decoding for t_y = sigmoid(p_y) + c_y
    if central_responsible:
        pred_y = pred_y + anchors_y
        pred_x = pred_x + anchors_x
    # decoding for t_y = c_h * p_y + c_y
    else:
        pred_y = anchors_h * pred_y + anchors_y
        pred_x = anchors_w * pred_x + anchors_x
    # decoding for t_h = a_h * exp(p_h)
    pred_h = anchors_h * tf.exp(pred_h)
    pred_w = anchors_w * tf.exp(pred_w)

    pred_ymin = pred_y - pred_h / 2.0
    pred_xmin = pred_x - pred_w / 2.0
    pred_ymax = pred_y + pred_h / 2.0
    pred_xmax = pred_x + pred_w / 2.0

    if clip:
        pred_ymin = tf.maximum(pred_ymin, 0)
        pred_xmin = tf.maximum(pred_xmin, 0)
        pred_ymax = tf.minimum(pred_ymax, 1)
        pred_xmax = tf.minimum(pred_xmax, 1)

    bboxes = tf.stack([pred_ymin, pred_xmin, pred_ymax, pred_xmax], axis=-1)
    return bboxes


def _layer_loss(locations, scores, encode_locations, encode_labels, encode_ious, pos_th, neg_th, neg_ratio,
                background_label=0, alpha=[1.0, 1.0, 1.0], HNM=False, batch_size=None):
    """
    Calculate loss for one layer,
    encode_labels corresponds to the GT box with highest iou, but this iou can be less than neg_th!
    so need to process and create new labels !
    :param locations: predicted locations [N, H, W, K, 4 ]
    :param scores: predicted scores [N, H, W, K, 21]
    :param encode_locations: [N, H, W, K, 4]
    :param encode_labels: [N, H, W, K]
    :param encode_ious: [N, H, W, K]
    :return:
    """
    positive_mask = encode_ious > pos_th

    # need to redefine the labels, those assgined to some class with iou < neg_th, should be assgined to background
    negative_mask = tf.logical_and(
        encode_ious <= neg_th,
        tf.logical_not(positive_mask)
    )
    # background_label for negative and label for positive
    negative_labels = tf.where(negative_mask,
                               background_label * tf.cast(negative_mask, tf.int32),
                               encode_labels)
    # tf.add_to_collection('debug', negative_labels)
    if batch_size is None:
        batch_size = tensor_shape(locations)[0]

    if HNM:
        positive_num = tf.reduce_sum(tf.cast(positive_mask, tf.int32))
        # calculate background scores
        neg_scores = tf.nn.softmax(scores, axis=-1)[..., background_label]
        neg_scores = tf.where(negative_mask,
                              neg_scores,
                              # set positive ones's negative score to be 1, so that it won't be count in top_k
                              1.0 - tf.cast(negative_mask, tf.float32)
                              )
        # solve #negative, add one so that neg_values has more than one value
        max_negative_num = tf.reduce_sum(tf.cast(negative_mask, tf.int32))

        negative_num = neg_ratio * positive_num + batch_size
        negative_num = tf.minimum(negative_num, max_negative_num)

        # Hard Negative Mining:
        # find those with lower background scores, but are indeed background!
        neg_values, _ = tf.nn.top_k(tf.reshape(-1.0 * neg_scores, [-1]), k=negative_num)
        negative_mask = tf.logical_and(
            negative_mask,
            neg_scores < -neg_values[-1]
        )

    positive_mask = tf.cast(positive_mask, tf.float32)
    negative_mask = tf.cast(negative_mask, tf.float32)

    tf.add_to_collection(ZERO_NUMBER_PER_LAYER_SCOPE, tf.reduce_sum(positive_mask * negative_mask))
    tf.add_to_collection(POS_NUMBER_PER_LAYER_SCOPE, tf.reduce_sum(positive_mask))
    tf.add_to_collection(NEG_NUMBER_PER_LAYER_SCOPE, tf.reduce_sum(negative_mask))

    with tf.name_scope('cross_entropy_loss'):
        with tf.name_scope('positive'):
            pos_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=encode_labels)
            pos_loss = tf.div(tf.reduce_sum(pos_loss * positive_mask), batch_size)
            pos_loss *= alpha[0]
            tf.add_to_collection(tf.GraphKeys.LOSSES, pos_loss)

        with tf.name_scope('negative'):
            neg_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=negative_labels)
            neg_loss = tf.div(tf.reduce_sum(neg_loss * negative_mask), batch_size)
            neg_loss *= alpha[1]
            tf.add_to_collection(tf.GraphKeys.LOSSES, neg_loss)

    with tf.name_scope('bbox_regression_loss'):
        bbox_loss = smooth_l1(locations - encode_locations)
        bbox_loss = tf.reduce_sum(bbox_loss, axis=-1)
        bbox_loss = tf.div(tf.reduce_sum(bbox_loss * positive_mask), batch_size)
        bbox_loss *= alpha[2]
        tf.add_to_collection(tf.GraphKeys.LOSSES, bbox_loss)

    return pos_loss, neg_loss, bbox_loss


def layers_predictions(end_points, num_classes, anchor_nums, feature_names, conv_params):
    """
    Gather predictions from layers
    :param end_points:
    :param num_classes:
    :return:
    """
    gather_locations, gather_scores = [], []
    for idx, key in enumerate(feature_names):
        layer = end_points[key]
        num_anchors = anchor_nums[idx]
        locations, scores = _layer_prediction(layer,
                                              num_anchors=num_anchors,
                                              num_classes=num_classes,
                                              conv_params=conv_params[idx],
                                              scope='feature2bbox{}'.format(idx + 1))
        gather_locations.append(locations)
        gather_scores.append(scores)
    return gather_locations, gather_scores


def layers_loss(prediction_gathers, encoding_gathers,
                pos_th=0.5, neg_th=0.3, neg_ratio=3, alpha=[1.0, 1.0, 1.0],
                HNM=False):
    gather_pred_locations, gather_pred_scores = prediction_gathers
    gather_truth_locations, gather_truth_labels, gather_truth_ious = encoding_gathers

    gather_pos_loss, gather_neg_loss, gather_bbox_loss = [], [], []
    for idx in range(len(gather_pred_locations)):
        pos_loss, neg_loss, bbox_loss = _layer_loss(
            locations=gather_pred_locations[idx],
            scores=gather_pred_scores[idx],
            encode_locations=gather_truth_locations[idx],
            encode_labels=gather_truth_labels[idx],
            encode_ious=gather_truth_ious[idx],
            pos_th=pos_th,
            neg_th=neg_th,
            neg_ratio=neg_ratio,
            alpha=alpha,
            HNM=HNM
        )
        gather_pos_loss.append(pos_loss)
        gather_neg_loss.append(neg_loss)
        gather_bbox_loss.append(bbox_loss)
    return gather_pos_loss, gather_neg_loss, gather_bbox_loss


def layers_loss_new(prediction_gathers, encoding_gathers,
                    pos_th=0.5, neg_th=0.3, neg_ratio=3,
                    alpha=[1.0, 1.0, 1.0], HNM=False):
    gather_pred_locations, gather_pred_scores = prediction_gathers
    gather_truth_locations, gather_truth_labels, gather_truth_ious = encoding_gathers

    concat_pred_locations = []
    concat_pred_scores = []
    concat_truth_locations = []
    concat_truth_labels = []
    concat_truth_ious = []

    batch_size = tensor_shape(gather_pred_scores[0])[0]
    num_classes = tensor_shape(gather_pred_scores[0])[-1]
    for idx in range(len(gather_pred_locations)):
        concat_pred_locations.append(tf.reshape(gather_pred_locations[idx], shape=[-1, 4]))
        concat_pred_scores.append(tf.reshape(gather_pred_scores[idx], shape=[-1, num_classes]))
        concat_truth_locations.append(tf.reshape(gather_truth_locations[idx], shape=[-1, 4]))
        concat_truth_labels.append(tf.reshape(gather_truth_labels[idx], shape=[-1]))
        concat_truth_ious.append(tf.reshape(gather_truth_ious[idx], shape=[-1]))

    concat_pred_locations = tf.concat(concat_pred_locations, axis=0)
    concat_pred_scores = tf.concat(concat_pred_scores, axis=0)
    concat_truth_locations = tf.concat(concat_truth_locations, axis=0)
    concat_truth_labels = tf.concat(concat_truth_labels, axis=0)
    concat_truth_ious = tf.concat(concat_truth_ious, axis=0)

    pos_loss, neg_loss, bbox_loss = _layer_loss(
        locations=concat_pred_locations,
        scores=concat_pred_scores,
        encode_locations=concat_truth_locations,
        encode_labels=concat_truth_labels,
        encode_ious=concat_truth_ious,
        pos_th=pos_th,
        neg_th=neg_th,
        neg_ratio=neg_ratio,
        batch_size=batch_size,
        alpha=alpha,
        HNM=HNM
    )
    return [pos_loss], [neg_loss], [bbox_loss]


def layers_select_nms(gather_pred_scores, gather_decode_bboxes, select_th=0.5, nms_th=0.45, soft_sigma=0.5, nms_k=200,
                      class_agnostic=True,
                      num_classes=21, softmax=True):
    """
    :param gather_pred_scores:
    :param gather_decode_bboxes:
    :param select_th:
    :param nms_th: if set to None, then using soft_nms mode
    :param soft_sigma:
    :param nms_k:
    :param num_classes:
    :return:
    """
    all_scores = []
    all_bboxes = []

    for idx in range(len(gather_pred_scores)):
        scores = gather_pred_scores[idx]
        # softmax here to put all comparisons in the same scale
        if softmax:
            scores = tf.reshape(tf.nn.softmax(scores, axis=-1), [-1, num_classes])
        else:
            # using binary cross entropy
            scores = tf.reshape(tf.nn.sigmoid(scores), [-1, num_classes])

        if class_agnostic:
            decode_bboxes = tf.reshape(gather_decode_bboxes[idx], [-1, 4])
        else:
            decode_bboxes = tf.reshape(gather_decode_bboxes[idx], [-1, num_classes, 4])
        print(scores, decode_bboxes)
        all_scores.append(scores)
        all_bboxes.append(decode_bboxes)

    # [None, N, 21]
    scores = tf.concat(all_scores, axis=0)
    # [None, N, 4]
    bboxes = tf.concat(all_bboxes, axis=0)
    print(scores, bboxes)

    gather_scores = {}
    gather_bboxes = {}
    class_scores_list = tf.unstack(scores, axis=-1)

    for class_id in range(num_classes):
        class_score = class_scores_list[class_id]
        if not class_agnostic:
            class_bboxes = clas

        if nms_th is not None:
            nms_idxes = tf.image.non_max_suppression(bboxes, class_score,
                                                     max_output_size=nms_k,
                                                     iou_threshold=nms_th)
            nms_score = tf.gather(class_score, nms_idxes)
            nms_boxes = tf.gather(bboxes, nms_idxes)
        else:
            nms_score, nms_boxes = soft_nms(class_score, bboxes, max_output_size=nms_k, sigma=soft_sigma)

        select_idxes = tf.squeeze(tf.where(nms_score >= select_th), axis=-1)
        select_score = tf.gather(nms_score, select_idxes)
        select_boxes = tf.gather(nms_boxes, select_idxes)

        gather_scores[class_id] = select_score
        gather_bboxes[class_id] = select_boxes

    return gather_scores, gather_bboxes


def layers_anchors(feat_shapes, anchors, offset=0.5):
    """
    Gather anchors from layers
    :param end_points:
    :return:
    """
    ys, xs, hs, ws = [], [], [], []
    for idx, shape in enumerate(feat_shapes):
        y, x, h, w = _layer_anchors(shape, anchors[idx], offset=offset)
        ys.append(y)
        xs.append(x)
        hs.append(h)
        ws.append(w)
    return ys, xs, hs, ws


def layers_encoding(all_anchors, labels, bboxes, background_label=0, prior_scale=None, central_responsible=False,
                    only_max=False):
    gather_locations, gather_labels, gather_ious = [], [], []
    ys, xs, hs, ws = all_anchors
    for idx in range(len(ys)):
        anchor = ys[idx], xs[idx], hs[idx], ws[idx]
        encode_locations, encode_labels, encode_ious = \
            _layer_encoding(anchor, labels, bboxes,
                            background_label=background_label,
                            prior_scale=prior_scale,
                            central_responsible=central_responsible,
                            only_max=only_max)
        gather_locations.append(encode_locations)
        gather_labels.append(encode_labels)
        gather_ious.append(encode_ious)
    return gather_locations, gather_labels, gather_ious


def layers_decoding(gather_locations, gather_anchors, clip=True, prior_scale=None, central_responsible=False):
    ys, xs, hs, ws = gather_anchors
    gather_decode_bboxes = []
    for idx in range(len(ys)):
        tmp = ys[idx], xs[idx], hs[idx], ws[idx]
        decode_bboxes = _layer_decode(gather_locations[idx], tmp, clip=clip, prior_scale=prior_scale,
                                      central_responsible=central_responsible)
        gather_decode_bboxes.append(decode_bboxes)
    return gather_decode_bboxes


def _yolo_detection_loss(locations, scores,
                         encode_locations, encode_labels, encode_ious,
                         pos_th, background_label=0, alpha=[1.0, 5.0, 1.0, 1.0]):
    """
    Calculate loss for one layer,
    encode_labels corresponds to the GT box with highest iou, but this iou can be less than neg_th!
    so need to process and create new labels !
    :param locations: predicted locations [N, H, W, K, 4 ]
    :param scores: predicted scores [N, H, W, K, 21]
    :param encode_locations: [N, H, W, K, 4]
    :param encode_labels: [N, H, W, K]
    :param encode_ious: [N, H, W, K]
    :return:
    """
    positive_mask = tf.logical_and(
        encode_labels != background_label,
        encode_ious > pos_th
    )
    positive_mask = tf.cast(positive_mask, tf.float32)
    tf.add_to_collection('positive_nums', tf.reduce_sum(positive_mask))
    num_classes = tensor_shape(scores)[-1]
    batch_size = tensor_shape(locations)[0]

    with tf.name_scope('classes_loss'):
        classes_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(tf.one_hot(encode_labels, depth=num_classes), dtype=tf.float32), logits=scores)
        classes_loss = classes_loss[..., 1:]
        classes_loss = tf.reduce_sum(classes_loss, axis=-1)
        classes_loss = alpha[0] * tf.reduce_sum(classes_loss * positive_mask) / batch_size
        tf.add_to_collection(tf.GraphKeys.LOSSES, classes_loss)

    with tf.name_scope('objectness_loss'):
        # use negative background as objectness
        object_scores = scores[..., background_label]
        tf.add_to_collection('objectness', object_scores)
        object_losses = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=positive_mask, logits=object_scores
        )
        is_obj_losses = alpha[1] * tf.reduce_sum(object_losses * positive_mask) / batch_size
        non_obj_losses = alpha[2] * tf.reduce_sum(object_losses * (1.0 - positive_mask)) / batch_size
        tf.add_to_collection(tf.GraphKeys.LOSSES, is_obj_losses)
        tf.add_to_collection(tf.GraphKeys.LOSSES, non_obj_losses)

    #
    with tf.name_scope('bbox_regression_loss'):
        bbox_loss = tf.square(locations - encode_locations)
        bbox_loss = tf.reduce_sum(bbox_loss, axis=-1)
        bbox_loss *= positive_mask
        bbox_loss = alpha[3] * tf.reduce_sum(bbox_loss) / batch_size
        tf.add_to_collection(tf.GraphKeys.LOSSES, bbox_loss)

    return classes_loss, is_obj_losses, non_obj_losses, bbox_loss


def yolo_detection_loss(locations, scores, encoding_gathers,
                        pos_th, background_label=0, alpha=[1.0, 5.0, 1.0, 1.0]):
    gather_truth_locations, gather_truth_labels, gather_truth_ious = encoding_gathers

    gather_classes_loss, gather_is_obj_losses, gather_non_obj_losses, gather_bbox_loss = [], [], [], []

    for idx in range(len(locations)):
        classes_loss, is_obj_losses, non_obj_losses, bbox_loss = _yolo_detection_loss(
            locations=locations[idx],
            scores=scores[idx],
            encode_locations=gather_truth_locations[idx],
            encode_labels=gather_truth_labels[idx],
            encode_ious=gather_truth_ious[idx],
            pos_th=pos_th,
            background_label=background_label,
            alpha=alpha
        )
        gather_classes_loss.append(classes_loss)
        gather_is_obj_losses.append(is_obj_losses)
        gather_non_obj_losses.append(non_obj_losses)
        gather_bbox_loss.append(bbox_loss)
    return gather_classes_loss, gather_is_obj_losses, gather_non_obj_losses, gather_bbox_loss
