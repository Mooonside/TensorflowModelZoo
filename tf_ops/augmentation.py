import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib.image.python.ops.image_ops import rotate
from tf_ops.detection_utils import bboxes_intersection, bboxs_clip, bboxes_resize
from tf_ops.wrap_ops import same_padding, tensor_shape, tensor_rank


def apply_random_op(op_func, kwargs, apply_prob=0.5):
    """
    Randomly apply op_func
    :param op_func: function
    :param kwargs: args of this function
    :return: op_func(args) or args
    """
    uniform_random = tf.random_uniform([], 0, 1.0)
    mirror_cond = tf.less(uniform_random, apply_prob)

    def identity_func(*x):
        return x

    return control_flow_ops.cond(mirror_cond, lambda: op_func(**kwargs), lambda: identity_func(*[kwargs[i] for i in kwargs.keys()]))


#####################################################################################
#                    Random Scale, Crop, Flip, ReShape                              #
#####################################################################################



def get_random_scale(min_scale_factor, max_scale_factor, step_size=0):
    """Gets a random scale value.

    Args:
      min_scale_factor: Minimum scale value.
      max_scale_factor: Maximum scale value.
      step_size: The step size from minimum to maximum value.

    Returns:
      A random scale value selected between minimum and maximum value.

    Raises:
      ValueError: min_scale_factor has unexpected value.
    """
    if min_scale_factor is None or max_scale_factor is None or step_size is None:
        return 1.0

    if min_scale_factor < 0 or min_scale_factor > max_scale_factor:
        raise ValueError('Unexpected value of min_scale_factor.')

    if min_scale_factor == max_scale_factor:
        return tf.to_float(min_scale_factor)

    # When step_size = 0, we sample the value uniformly from [min, max).
    if step_size == 0:
        return tf.random_uniform([],
                                 minval=min_scale_factor,
                                 maxval=max_scale_factor)

    # When step_size != 0, we randomly select one discrete value from [min, max].
    num_steps = int((max_scale_factor - min_scale_factor) / step_size + 1)
    scale_factors = tf.lin_space(min_scale_factor, max_scale_factor, num_steps)
    shuffled_scale_factors = tf.random_shuffle(scale_factors)
    return shuffled_scale_factors[0]


def scale_image(image, scale):
    """Randomly scales image

    Args:
      image: Image with shape [height, width, 3].
      scale: The value to scale image and label.

    Returns:
      Scaled image and label.
    """
    if scale == 1.0:
        return image
    if type(image) is list:
        return [ scale_image(i, scale) for i in image]

    assert image.dtype is tf.float32

    image_shape = tf.shape(image)
    new_dim = tf.to_int32(tf.to_float([image_shape[0], image_shape[1]]) * scale)

    # Need squeeze and expand_dims because image interpolation takes
    # 4D tensors as input.
    image = tf.expand_dims(image, axis=0)
    image = tf.image.resize_bilinear(image, new_dim, align_corners=True)
    image = tf.squeeze(image, axis=0)

    return image


def scale_segmentation(segmentation, scale):
    """Randomly scales image

    Args:
      image: Image with shape [height, width, 3].
      scale: The value to scale image and label.

    Returns:
      Scaled image and label.
    """
    if scale == 1.0:
        return segmentation
    if type(segmentation) is list:
        return [ scale_segmentation(i, scale) for i in segmentation]

    assert segmentation.dtype in [tf.int32, tf.int64, tf.uint8, tf.uint16, tf.int8, tf.int16]
    segmentation_shape = tf.shape(segmentation)
    segmentation_type = segmentation.dtype
    # TODO: first cast to int32
    segmentation = tf.cast(segmentation, tf.int32)
    new_dim = tf.to_int32(tf.to_float([segmentation_shape[0], segmentation_shape[1]]) * scale)

    # Need squeeze and expand_dims because image interpolation takes 4D tensors as input.
    segmentation = tf.expand_dims(segmentation, axis=0)
    segmentation = tf.image.resize_nearest_neighbor(segmentation, new_dim, align_corners=True)
    segmentation = tf.squeeze(segmentation, axis=0)
    segmentation = tf.cast(segmentation, segmentation_type)

    return segmentation


def random_scale_within_aspect_ratio(image, min_scale, max_scale, min_ratio, max_ratio):
    image_h, image_w = tf.shape(image)[0], tf.shape(image)[1]
    random_scale = get_random_scale(min_scale, max_scale)
    random_aspect_ratio = get_random_scale(min_ratio, max_ratio)

    crop_h = tf.minimum(
        tf.cast(
            tf.sqrt(random_scale * random_aspect_ratio) * tf.cast(image_h, tf.float32), tf.int32), image_h
    )
    crop_w = tf.minimum(
        tf.cast(
            tf.sqrt(random_scale / random_aspect_ratio) * tf.cast(image_w, tf.float32), tf.int32), image_w
    )

    crop_bbox = generate_random_box(image_h, image_w, crop_h, crop_w)
    crop_image = crop_image_by_box(image, crop_bbox)
    return crop_image, crop_bbox


def generate_random_box(max_y, max_x, box_height, box_width):
    """
    Generates a random box of size [box_height, box_width] within a playground of maximum size [max_y, max_x]
    :param image:
    :param crop_height:
    :param crop_width:
    :return:
    """
    crop_size_assert = tf.Assert(
        tf.logical_and(
            tf.greater_equal(max_y, box_height),
            tf.greater_equal(max_x, box_width)),
        ['Crop size greater than the image size.'])

    # Create a random bounding box.
    # Use tf.random_uniform and not numpy.random.rand as doing the former would
    # generate random numbers at graph eval time, unlike the latter which
    # generates random numbers at graph definition time.
    with tf.control_dependencies([crop_size_assert]):
        max_offset_height = max_y - box_height + 1
        max_offset_width = max_x - box_width + 1
    offset_height = tf.random_uniform(
        [], maxval=max_offset_height, dtype=tf.int32)
    offset_width = tf.random_uniform(
        [], maxval=max_offset_width, dtype=tf.int32)

    box = [offset_height, offset_width, offset_height + box_height, offset_width + box_width]
    return box


def crop_image_by_box(image, crop_box):
    """Crops the given image.

    Args:
      image: a image tensor [H, W, 3]
      crop_box: the crop box

    Returns:
      the cropped image
    """
    if type(image) is list:
        return [crop_image_by_box(i, crop_box) for i in image]

    ymin, xmin, ymax, xmax = crop_box
    offsets = tf.to_int32(tf.stack([ymin, xmin, 0]))
    cropped_shape = tf.stack([ymax - ymin, xmax - xmin, tf.shape(image)[2]])

    crop_image = tf.slice(image, offsets, cropped_shape)
    crop_image = tf.reshape(crop_image, cropped_shape)

    return crop_image


def pad_values_eqaully(image, value, target_height, target_width):
    """pad values around image to make it reach target_height and target_width

    Args:
      image: a image tensor [H, W, C]
      value: value padded [C]
      target_height: the new height.
      target_width: the new width.

    Returns:
      the padded image
    """
    dtype = image.dtype

    image = tf.cast(image, tf.float32)
    image -= value

    image_shape = tf.shape(image)
    height, width = image_shape[0], image_shape[1]

    pad_heights = target_height - height
    pad_widths = target_width - width

    height_params = tf.stack([pad_heights // 2, pad_heights - pad_heights // 2])
    width_params = tf.stack([pad_widths // 2, pad_widths - pad_widths // 2])
    channel_params = tf.stack([0, 0])
    # [3, 2]
    paddings = tf.stack([height_params, width_params, channel_params])
    pad_image = tf.pad(image, paddings, constant_values=0)
    pad_image += value
    pad_image = tf.cast(pad_image, dtype)

    return pad_image


def reshape_image(image, target_height, target_width, pad_value=None, keep_aspect_ratio=False,
                  method=tf.image.resize_bilinear):
    image_shape = tf.shape(image)
    height, width = image_shape[0], image_shape[1]

    if keep_aspect_ratio:
        # judge which side will reach target size first
        scale_height = target_height / height
        scale_width = target_width / width
        scale = tf.cast(tf.minimum(scale_height, scale_width), tf.float32)
        image = scale_image(image, scale)
        # pad values
        image = pad_values_eqaully(image, pad_value, target_height, target_width)
    else:
        # simply reshape
        image = tf.expand_dims(image, axis=0)
        image = method(image, [target_height, target_width], align_corners=True)
        image = tf.squeeze(image, axis=0)
    return image


def flip_bboxes_left_right(bboxes):
    """
    Flip bounding boxes coordinates.
    bboxes : [N, 4] , 4 as ymins, xmins, ymaxs, xmaxs
    """
    bboxes = tf.stack([bboxes[:, 0], 1 - bboxes[:, 3],
                       bboxes[:, 2], 1 - bboxes[:, 1]], axis=-1)
    return bboxes


def flip_image_left_right(image):
    """
    Flip Image From Left To Right
    :param image: [H, W, C]
    :return:
    """
    return tf.reverse(image, axis=[1])


def flip_left_right(images, bboxes=None):
    if type(images) is list:
        flipped_images = [flip_image_left_right(image) for image in images]
    else:
        flipped_images = flip_image_left_right(images)

    if bboxes is not None:
        bboxes = flip_bboxes_left_right(bboxes)
        return flipped_images, bboxes
    else:
        return flipped_images


def distorted_bounding_box_crop(image, bboxes_labels, bboxes,
                                min_object_covered=0.3,
                                aspect_ratio_range=(0.9, 1.1),
                                area_range=(0.1, 1.0),
                                max_attempts=200,
                                intersection_threshold=0.5,
                                clip=True,
                                scope=None):
    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bboxes]):
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].
        bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=tf.expand_dims(bboxes, 0),
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True)
        # [y_min, x_min, y_max, x_max]
        distort_bbox = distort_bbox[0, 0]

        # Crop the image to the specified bounding box.
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        # Restore the shape since the dynamic slice loses 3rd dimension.
        cropped_image.set_shape([None, None, 3])

        # Update bounding boxes: resize and filter out.
        bboxes = bboxes_resize(bboxes, distort_bbox)

        intersections = bboxes_intersection(bboxes, tf.constant([0, 0, 1, 1], bboxes.dtype))
        mask = intersections > intersection_threshold
        labels = tf.boolean_mask(bboxes_labels, mask)
        bboxes = tf.boolean_mask(bboxes, mask)

        if clip:
            bboxes = bboxs_clip(bboxes, vmin=0, vmax=1)

        ymin, xmin = bbox_begin[0], bbox_begin[1]
        ymax, xmax = ymin + bbox_size[0], xmin + bbox_size[1]
        crop_box = [ymin, xmin, ymax, xmax]
    return cropped_image, labels, bboxes, crop_box


#####################################################################################
#                                  Color Distortion                                 #
#####################################################################################
def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
    """Distort the color of a Tensor image.

    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.

    Args:
        image: 3-D Tensor containing single image in [0, 1].
        color_ordering: Python int, a type of distortion (valid values: 0-3).
        fast_mode: Avoids slower ops (random_hue and random_contrast)
        scope: Optional scope for name_scope.
    Returns:
        3-D Tensor color-distorted image on range [0, 1]
    Raises:
        ValueError: if color_ordering not in [0, 3]
    """
    with tf.name_scope(scope, 'distort_color', [image]):
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32.)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32.)
            else:
                raise ValueError('color_ordering must be in [0, 3]')
        # The random_* ops do not necessarily clamp.
        image = tf.clip_by_value(image, 0, 255)
        return image


#####################################################################################
#                                  Rotation                                         #
#####################################################################################
def random_rotate(image, segment, pad_value, ignore_label, min_degree=-10, max_degree=10):
    """
    :param image: [H, W, 3]
    :param segment:  [H, W, 1] or [H, W]
    :param pad_value:
    :param ignore_label:
    :param min_degree:
    :param max_degree:
    :return:
    """
    if segment is None:
        return _random_rotate_img(image, pad_value, min_degree=-min_degree, max_degree=max_degree)

    angle = tf.random_uniform(shape=[], minval=min_degree, maxval=max_degree)
    angle = angle / 180 * 3.1415926
    if tensor_rank(segment) == 2:
        # to support mask's broadcast
        segment = tf.expand_dims(segment, axis=-1)
    image_dtype, segment_dtype = image.dtype, segment.dtype

    image = tf.cast(image, tf.float32)
    segment = tf.cast(segment, tf.int32)
    ones_mask = tf.ones_like(segment, dtype=tf.int32)

    rot_img = rotate(image, angle, interpolation='BILINEAR')
    rot_segment = rotate(segment, angle, interpolation='NEAREST')
    rot_ones_mask = rotate(ones_mask, angle, interpolation='NEAREST')

    float_mask = tf.cast(rot_ones_mask, tf.float32)
    # float_mask's last dimeension is one, so will do broadcast!
    rot_img = float_mask * rot_img + (1 - float_mask) * pad_value

    int_mask = tf.cast(rot_ones_mask, tf.int32)

    rot_segment = int_mask * rot_segment + (1 - int_mask) * ignore_label

    rot_img = tf.cast(rot_img, image_dtype)
    rot_segment = tf.cast(rot_segment, segment_dtype)
    return rot_img, rot_segment


def _random_rotate_img(image, pad_value, min_degree=-10, max_degree=10):
    '''
    :param image: [H, W, 3]
    :param pad_value:
    :param min_degree:
    :param max_degree:
    :return:
    '''
    angle = tf.random_uniform(shape=[], minval=min_degree, maxval=max_degree)
    angle = angle / 180 * 3.1415926

    image_dtype = image.dtype
    image = tf.cast(image, tf.float32)
    ones_mask = tf.ones_like(image, dtype=tf.int32)[..., 0]

    rot_img = rotate(image, angle, interpolation='BILINEAR')
    rot_ones_mask = rotate(ones_mask, angle, interpolation='NEAREST')
    float_mask = tf.cast(rot_ones_mask, tf.float32)
    # float_mask's last dimeension is one, so will do broadcast!
    rot_img = float_mask * rot_img + (1 - float_mask) * pad_value

    rot_img = tf.cast(rot_img, image_dtype)
    return rot_img


#####################################################################################
#                                  Gaussian Blur                                    #
#####################################################################################
def gaussian_blur(img, kernel_size=5, sigma=1.0):
    """
    :param img: [H, W, C] or [N, H, W, C]
    :param kernel_size:
    :param sigma:
    :return:
    """
    if type(kernel_size) is list:
        kernel_size = tf.random_shuffle(kernel_size)[0]

    tf.add_to_collection('kernel_size', kernel_size)
    squeeze = False
    if tensor_rank(img) == 3:
        img = tf.expand_dims(img, axis=0)
        squeeze = True

    # generate gaussian kernel
    g_r = tf.range(kernel_size)

    kernel_size = tf.cast(kernel_size, tf.float32)
    g_r = tf.cast(g_r, tf.float32)

    if sigma is None:
    # https://docs.opencv.org/3.1.0/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa
        sigma = 0.3 *((kernel_size - 1)* 0.5 - 1) + 0.8

    g_r = tf.exp(-1.0 * (g_r - 0.5 * (kernel_size - 1)) ** 2 / (2.0 * sigma ** 2))
    g_r = g_r / tf.reduce_sum(g_r)
    g_2d = g_r[tf.newaxis, ...] * g_r[..., tf.newaxis]
    g_2d = g_2d[..., tf.newaxis, tf.newaxis]

    kernel_size = tf.cast(kernel_size, tf.int32)
    f = lambda x : tf.nn.conv2d(
        same_padding(x, [kernel_size, kernel_size], [1, 1]),
        filter=g_2d,
        strides=[1, 1, 1, 1],
        padding='VALID',
    )

    blurs = []
    for i in range(tensor_shape(img)[-1]):
        blurs.append(f(img[..., i][..., tf.newaxis]))

    blur = tf.concat(blurs, axis=-1)
    if squeeze:
        blur = tf.squeeze(blur, axis=0)
    return blur