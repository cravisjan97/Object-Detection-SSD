# Copyright 2015 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Pre-processing images for SSD-type networks.
"""
from enum import Enum, IntEnum
import numpy as np

import tensorflow as tf
import tf_extended as tfe

from tensorflow.python.ops import control_flow_ops

from preprocessing import tf_image
# from nets import ssd_common

slim = tf.contrib.slim

# Resizing strategies.
Resize = IntEnum('Resize', ('NONE',                # Nothing!
                            'CENTRAL_CROP',        # Crop (and pad if necessary).
                            'PAD_AND_RESIZE',      # Pad, and resize to output shape.
                            'WARP_RESIZE'))        # Warp resize.

# VGG mean parameters.
_R_MEAN = 123.
_G_MEAN = 117.
_B_MEAN = 104.

# Some training pre-processing parameters.
BBOX_CROP_OVERLAP = 0.4        # Minimum overlap to keep a bbox after cropping.
CROP_RATIO_RANGE = (0.8, 1.2)  # Distortion ratio during cropping.
EVAL_SIZE = (300, 300)


def tf_image_whitened(image, means=[_R_MEAN, _G_MEAN, _B_MEAN]):
    """Subtracts the given means from each image channel.

    Returns:
        the centered image.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    mean = tf.constant(means, dtype=image.dtype)
    image = image - mean
    return image


def tf_image_unwhitened(image, means=[_R_MEAN, _G_MEAN, _B_MEAN], to_int=True):
    """Re-convert to original image distribution, and convert to int if
    necessary.

    Returns:
      Centered image.
    """
    mean = tf.constant(means, dtype=image.dtype)
    image = image + mean
    if to_int:
        image = tf.cast(image, tf.int32)
    return image


def np_image_unwhitened(image, means=[_R_MEAN, _G_MEAN, _B_MEAN], to_int=True):
    """Re-convert to original image distribution, and convert to int if
    necessary. Numpy version.

    Returns:
      Centered image.
    """
    img = np.copy(image)
    img += np.array(means, dtype=img.dtype)
    if to_int:
        img = img.astype(np.uint8)
    return img


def tf_summary_image(image, bboxes, name='image', unwhitened=False):
    """Add image with bounding boxes to summary.
    """
    if unwhitened:
        image = tf_image_unwhitened(image)
    image = tf.expand_dims(image, 0)
    bboxes = tf.expand_dims(bboxes, 0)
    image_with_box = tf.image.draw_bounding_boxes(image, bboxes)
    tf.summary.image(name, image_with_box)


def apply_with_random_selector(x, func, num_cases):
    """Computes func(x, sel), with sel sampled from [0...num_cases-1].

    Args:
        x: input Tensor.
        func: Python function to apply.
        num_cases: Python int32, number of cases to sample sel from.

    Returns:
        The result of func(x, sel), where func receives the value of the
        selector as a python integer, but sel is sampled dynamically.
    """
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge([
            func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
            for case in range(num_cases)])[0]


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
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
            else:
                raise ValueError('color_ordering must be in [0, 3]')
        # The random_* ops do not necessarily clamp.
        return tf.clip_by_value(image, 0.0, 1.0)


def distorted_bounding_box_crop(image,
                                labels,
                                bboxes,
                                min_object_covered=0.25,
                                aspect_ratio_range=(0.5, 2.0),
                                area_range=(0.1, 1.0),
                                max_attempts=200,
                                bbox_crop_overlap = 0.5,
                                scope=None):
    """Generates cropped_image using a one of the bboxes randomly distorted.

    See `tf.image.sample_distorted_bounding_box` for more documentation.

    Args:
        image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
        bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
            where each coordinate is [0, 1) and the coordinates are arranged
            as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
            image.
        min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
            area of the image must contain at least this fraction of any bounding box
            supplied.
        aspect_ratio_range: An optional list of `floats`. The cropped area of the
            image must have an aspect ratio = width / height within this range.
        area_range: An optional list of `floats`. The cropped area of the image
            must contain a fraction of the supplied image within in this range.
        max_attempts: An optional `int`. Number of attempts at generating a cropped
            region of the image of the specified constraints. After `max_attempts`
            failures, return the entire image.
        scope: Optional scope for name_scope.
    Returns:
        A tuple, a 3-D Tensor cropped_image and the distorted bbox
    """
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
        distort_bbox = distort_bbox[0, 0]

        # Crop the image to the specified bounding box.
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        # Restore the shape since the dynamic slice loses 3rd dimension.
        cropped_image.set_shape([None, None, 3])

        # Update bounding boxes: resize and filter out.
        bboxes = tfe.bboxes_resize(distort_bbox, bboxes)
        labels, bboxes = tfe.bboxes_filter_overlap(labels, bboxes,
                                                   bbox_crop_overlap)
        
        #adjust bbbox coordinates so that they go back to [0,1]
        #this is essentially keeping only the overlapped of the bbox
        
        ymin = tf.maximum(bboxes[:,0], 0.0)
        
        xmin = tf.maximum(bboxes[:,1], 0.0)
        ymax = tf.minimum(bboxes[:,2], 1.0)
        xmax = tf.minimum(bboxes[:,3], 1.0)
        bboxes = tf.stack([ymin,xmin,ymax,xmax], axis = -1)

        return cropped_image, labels, bboxes, distort_bbox

def scale_randomly_image_with_annotation_with_fixed_size_output(img_tensor,
                                                                annotation_tensor,
                                                                output_shape,
                                                                min_relative_random_scale_change=0.8,
                                                                max_realtive_random_scale_change=1.2,
                                                                mask_out_number=255):
    """Returns tensor of a size (output_shape, output_shape, depth) and (output_shape, output_shape, 1).
    The function returns tensor that is of a size (output_shape, output_shape, depth)
    which is randomly scaled by a factor that is sampled from a uniform distribution
    between values [min_relative_random_scale_change, max_realtive_random_scale_change] multiplied
    by the factor that is needed to scale image to the output_shape. When the rescaled image
    doesn't fit into the [output_shape] size, the image is either padded or cropped. Also, the
    function returns scaled annotation tensor of the size (output_shape, output_shape, 1). Both,
    the image tensor and the annotation tensor are scaled using nearest neighbour interpolation.
    This was done to preserve the annotation labels. Be careful when specifying the big sample
    space for the random variable -- aliasing effects can appear. When scaling, this function
    preserves the aspect ratio of the original image. When performing all of those manipulations
    there will be some regions in the output image with blank regions -- the function masks out
    those regions in the annotation using mask_out_number. Overall, the function performs the
    rescaling neccessary to get image of output_shape, adds random scale jitter, preserves
    scale ratio, masks out unneccassary regions that appear.
    
    Parameters
    ----------
    img_tensor : Tensor of size (width, height, depth)
        Tensor with image
    annotation_tensor : Tensor of size (width, height, 1)
        Tensor with respective annotation
    output_shape : Tensor or list [int, int]
        Tensor of list representing desired output shape
    min_relative_random_scale_change : float
        Lower bound for uniform distribution to sample from
        when getting random scaling jitter
    max_realtive_random_scale_change : float
        Upper bound for uniform distribution to sample from
        when getting random scaling jitter
    mask_out_number : int
        Number representing the mask out value.
        
    Returns
    -------
    cropped_padded_img : Tensor of size (output_shape[0], output_shape[1], 3).
        Image Tensor that was randomly scaled
    cropped_padded_annotation : Tensor of size (output_shape[0], output_shape[1], 1)
        Respective annotation Tensor that was randomly scaled with the same parameters
    """
    
    # tf.image.resize_nearest_neighbor needs
    # first dimension to represent the batch number
    img_batched = tf.expand_dims(img_tensor, 0)
    annotation_batched = tf.expand_dims(annotation_tensor, 0)

    # Convert to int_32 to be able to differentiate
    # between zeros that was used for padding and
    # zeros that represent a particular semantic class
    annotation_batched = tf.to_int32(annotation_batched)

    # Get height and width tensors
    input_shape = tf.shape(img_batched)[1:3]

    input_shape_float = tf.to_float(input_shape)

    scales = output_shape / input_shape_float

    rand_var = tf.random_uniform(shape=[1],
                                 minval=min_relative_random_scale_change,
                                 maxval=max_realtive_random_scale_change)

    final_scale = tf.reduce_min(scales) * rand_var

    scaled_input_shape = tf.to_int32(tf.round(input_shape_float * final_scale))
    
    # Resize the image and annotation using nearest neighbour
    # Be careful -- may cause aliasing.
    
    # TODO: try bilinear resampling for image only
    resized_img = tf.image.resize_nearest_neighbor( img_batched, scaled_input_shape )
    resized_annotation = tf.image.resize_nearest_neighbor( annotation_batched, scaled_input_shape )

    resized_img = tf.squeeze(resized_img, axis=0)
    resized_annotation = tf.squeeze(resized_annotation, axis=0)

    # Shift all the classes by one -- to be able to differentiate
    # between zeros representing padded values and zeros representing
    # a particular semantic class.
    annotation_shifted_classes = resized_annotation + 1

    cropped_padded_img = tf.image.resize_image_with_crop_or_pad( resized_img, output_shape[0], output_shape[1] )

    cropped_padded_annotation = tf.image.resize_image_with_crop_or_pad(annotation_shifted_classes,
                                                                       output_shape[0],
                                                                       output_shape[1])
    
    # TODO: accept the classes lut instead of mask out
    # value as an argument
    annotation_additional_mask_out = tf.to_int32(tf.equal(cropped_padded_annotation, 0)) * (mask_out_number+1)

    cropped_padded_annotation = cropped_padded_annotation + annotation_additional_mask_out - 1
    
return cropped_padded_img, cropped_padded_annotation


def preprocess_for_train(image, labels, bboxes,
                         out_shape=EVAL_SIZE, data_format='NHWC',
                         scope='ssd_preprocessing_train'):
    """Preprocesses the given image for training.

    Note that the actual resizing scale is sampled from
        [`resize_size_min`, `resize_size_max`].

    Args:
        image: A `Tensor` representing an image of arbitrary size.
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.
        resize_side_min: The lower bound for the smallest side of the image for
            aspect-preserving resizing.
        resize_side_max: The upper bound for the smallest side of the image for
            aspect-preserving resizing.

    Returns:
        A preprocessed image.
    """
    fast_mode = False
    with tf.name_scope(scope, 'ssd_preprocessing_train', [image, labels, bboxes]):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')
        # Convert to float scaled [0, 1].
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        tf_summary_image(image, bboxes, 'original_image')

        # # Remove DontCare labels.
        # labels, bboxes = ssd_common.tf_bboxes_filter_labels(out_label,
        #                                                     labels,
        #                                                     bboxes)

        # Distort image and bounding boxes.
        dst_image = image
        dst_image, labels, bboxes, distort_bbox = \
            distorted_bounding_box_crop(dst_image, labels, bboxes)
             
        tf_summary_image(image, tf.reshape(distort_bbox, (1,-1)), 'cropped_position')

	# Random scaling of an image
	dst_image = scale_randomly_image_with_annotation_with_fixed_size_output(dst_image, bboxes, out_shape)
        
        # Resize image to output size.
        dst_image = tf_image.resize_image(dst_image, out_shape,
                                          method=tf.image.ResizeMethod.BILINEAR,
                                          align_corners=False)
        # Randomly flip the image horizontally.
        dst_image, bboxes = tf_image.random_flip_left_right(dst_image, bboxes)
        tf_summary_image(dst_image, bboxes, 'resized_image')

        # Randomly distort the colors. There are 4 ways to do it.
        dst_image = apply_with_random_selector(
                dst_image,
                lambda x, ordering: distort_color(x, ordering, fast_mode),
                num_cases=4)
        tf_summary_image(dst_image, bboxes, 'color_distorted_image')

        # Rescale to VGG input scale.
        image = dst_image * 255.
        image = tf_image_whitened(image, [_R_MEAN, _G_MEAN, _B_MEAN])
        # Image data format.
        if data_format == 'NCHW':
            image = tf.transpose(image, perm=(2, 0, 1))
        return image, labels, bboxes


def preprocess_for_eval(image, labels, bboxes,
                        out_shape=EVAL_SIZE, data_format='NHWC',
                        difficults=None, resize=Resize.WARP_RESIZE,
                        scope='ssd_preprocessing_train'):
    """Preprocess an image for evaluation.

    Args:
        image: A `Tensor` representing an image of arbitrary size.
        out_shape: Output shape after pre-processing (if resize != None)
        resize: Resize strategy.

    Returns:
        A preprocessed image.
    """
    with tf.name_scope(scope):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')

        image = tf.to_float(image)
        image = tf_image_whitened(image, [_R_MEAN, _G_MEAN, _B_MEAN])

        # Add image rectangle to bboxes.
        bbox_img = tf.constant([[0., 0., 1., 1.]])
        if bboxes is None:
            bboxes = bbox_img
        else:
            bboxes = tf.concat([bbox_img, bboxes], axis=0)

        if resize == Resize.NONE:
            # No resizing...
            pass
        elif resize == Resize.CENTRAL_CROP:
            # Central cropping of the image.
            image, bboxes = tf_image.resize_image_bboxes_with_crop_or_pad(
                image, bboxes, out_shape[0], out_shape[1])
        elif resize == Resize.PAD_AND_RESIZE:
            # Resize image first: find the correct factor...
            shape = tf.shape(image)
            factor = tf.minimum(tf.to_double(1.0),
                                tf.minimum(tf.to_double(out_shape[0] / shape[0]),
                                           tf.to_double(out_shape[1] / shape[1])))
            resize_shape = factor * tf.to_double(shape[0:2])
            resize_shape = tf.cast(tf.floor(resize_shape), tf.int32)

            image = tf_image.resize_image(image, resize_shape,
                                          method=tf.image.ResizeMethod.BILINEAR,
                                          align_corners=False)
            # Pad to expected size.
            image, bboxes = tf_image.resize_image_bboxes_with_crop_or_pad(
                image, bboxes, out_shape[0], out_shape[1])
        elif resize == Resize.WARP_RESIZE:
            # Warp resize of the image.
            image = tf_image.resize_image(image, out_shape,
                                          method=tf.image.ResizeMethod.BILINEAR,
                                          align_corners=False)

        # Split back bounding boxes.
        bbox_img = bboxes[0]
        bboxes = bboxes[1:]
        # Remove difficult boxes.
        if difficults is not None:
            mask = tf.logical_not(tf.cast(difficults, tf.bool))
            labels = tf.boolean_mask(labels, mask)
            bboxes = tf.boolean_mask(bboxes, mask)
        # Image data format.
        if data_format == 'NCHW':
            image = tf.transpose(image, perm=(2, 0, 1))
        return image, labels, bboxes, bbox_img


def preprocess_image(image,
                     labels,
                     bboxes,
                     out_shape,
                     data_format,
                     is_training=False,
                     **kwargs):
    """Pre-process an given image.

    Args:
      image: A `Tensor` representing an image of arbitrary size.
      output_height: The height of the image after preprocessing.
      output_width: The width of the image after preprocessing.
      is_training: `True` if we're preprocessing the image for training and
        `False` otherwise.
      resize_side_min: The lower bound for the smallest side of the image for
        aspect-preserving resizing. If `is_training` is `False`, then this value
        is used for rescaling.
      resize_side_max: The upper bound for the smallest side of the image for
        aspect-preserving resizing. If `is_training` is `False`, this value is
         ignored. Otherwise, the resize side is sampled from
         [resize_size_min, resize_size_max].

    Returns:
      A preprocessed image.
    """
    if is_training:
        return preprocess_for_train(image, labels, bboxes,
                                    out_shape=out_shape,
                                    data_format=data_format)
    else:
        return preprocess_for_eval(image, labels, bboxes,
                                   out_shape=out_shape,
                                   data_format=data_format,
                                   **kwargs)
