# author @Mrutyunnjay (mrutyunjay.tutulu1021@gmail.com)
"""
Utility functions for network skeleton and other functionality.
"""
import numpy as np
import tensorflow as tf


def iou(box1, box2):
    """
    computes iou of two boxes.
    :param box1: cx, cy, w, h
    :param box2: cx, cy, w, h
    :return: iou of box1 and box2
    """
    # check for x coordinate
    xc = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])

    # if intersecting :
    if xc > 0:
        # check for y coordinates
        yc = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - max(box1[1] - 0.5 * box1[3],
                                                                         box2[1] - 0.5 * box2[3])
        # if intersecting :
        if yc > 0:
            intersection = yc * xc
            union = box1[2] * box1[3] + box2[2] * box2[3] - intersection
            # box1 area + box2 area + intersection area

            return intersection / union
    return 0


def batch_iou(boxes, box):
    """
    compute the IOU of a batch of boxes with gt or another box
    :param boxes: 2D array with shape [cx, cy, w, h] and no of boxes similar to batch size
    :param box: single box as prev
    :return: iou score
    """
    # implement similar to prev function, just scale for multiple boxes
    xc = np.maximum(
        np.minimum(boxes[:, 0] + 0.5 * boxes[:, 2], box[0] + 0.5 * box[2]) - np.maximum(boxes[:, 0] - 0.5 * boxes[:, 2],
                                                                                        box[0] - 0.5 * box[2]),
        0
    )
    yc = np.maximum(
        np.minimum(boxes[:, 1] + 0.5 * boxes[:, 3], box[1] + 0.5 * box[3]) - np.maximum(boxes[:, 1] - 0.5 * boxes[:, 3],
                                                                                        box[1] - 0.5 * box[3]),
        0
    )

    intersection = xc * yc
    union = boxes[:, 2] * boxes[:, 3] + box[2] * box[3] - intersection

    return intersection / union


def nms(boxes, probs, threshold):
    """
    Performs Non-max Suppression.
    href : https://github.com/liweiac/fire-FRD-CNN/blob/master/src/utils/util.py
    :param boxes: array of [cx, cy, w, h] i.e detected box shapes
    :param probs: probs score of each boxes
    :param threshold: iou threshold or prob threshold for each boxes
    :return: keep true or false for corresponding boxes satisfying the criteria
    """
    # grab the sorted indexes from our array containing prob for each boxes
    order = probs.argsort()[::-1]
    # initialize a response array containing the declaration for each boxes
    # initialize by True by default
    keep = [True] * len(order)

    for i in range(len(order) - 1):
        overlaps = batch_iou(boxes[order[i + 1:]],
                             boxes[order[i]])  # calculate overlaps for each box corresponding to the rest

        for j, overlap in enumerate(overlaps):
            if overlap > threshold:
                keep[order[j + i + 1]] = False

    return keep


def bgr_to_rgb(images):
    """
    convert list of images from BGR to RGB
    :param images:
    :return:
    """
    outImages = []
    for image in images:
        outImages.append(image[:, :, ::-1])

    return outImages


def bbox_transform(bbox):
    """
    convert a boundary box of form [cx, cy, w, h] to [xmin, ymin, xmax, ymax] i.e diagonal coords.
    compatible with np array or tensors
    :param bbox:
    :return:
    """
    with tf.compat.v1.variable_scope('bbox_transform') as scope:
        cx, cy, w, h = bbox  # extract from given format
        outBox = [[]] * 4
        outBox[0] = cx - w / 2
        outBox[1] = cy - h / 2
        outBox[2] = cx + w / 2
        outBox[3] = cy + h / 2

    return outBox


def bbox_transform_inv(bbox):
    """
    convert from [xmin, ymin, xmax, ymax] format to [cx, cy, w, h]
    :param bbox:
    :return:
    """
    with tf.compat.v1.variable_scope('bbox_transform_inv') as scope:
        xmin, ymin, xmax, ymax = bbox
        outBox = [[]] * 4

        w = xmax - xmin + 1
        h = ymax - ymin + 1
        outBox[0] = xmin + 0.5 * w
        outBox[1] = ymin + 0.5 * h
        outBox[2] = w
        outBox[3] = h

    return outBox


def safe_exp(w, thresh):
    """Safe exponential function for tensors."""

    slope = np.exp(thresh)
    with tf.compat.v1.variable_scope('safe_exponential'):
        lin_bool = w > thresh
        lin_region = tf.compat.v1.to_float(lin_bool)

        lin_out = slope * (w - thresh + 1.)
        exp_out = tf.exp(tf.where(lin_bool, tf.zeros_like(w), w))

        out = lin_region * lin_out + (1. - lin_region) * exp_out
    return out


def sparse_to_dense(sp_indices, output_shape, values, default_value=0):
    """
    A dense Mat from sparse one.
    :param sp_indices: array containing index to place values
    :param output_shape: shape of the dense desired mat
    :param values: values corresponds to index in each row of sparse mat
    :param default_value: for non-specified indices
    :return: dense ndarray
    """

