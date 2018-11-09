# -------------------------------------------------------------------------
# MSDS R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Chengyang Li, based on code from Ross Girshick and Xinlei Chen
# -------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
try:
  import cPickle as pickle
except ImportError:
  import pickle

from utils.blob import im_list_to_blob

from model.config import cfg

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im1_orig = im[0].astype(np.float32, copy=True)
  im1_orig -= cfg.PIXEL_MEANS

  im2_orig = im[1].astype(np.float32, copy=False)
  im2_orig -= cfg.PIXEL_MEANS

  im_shape = im1_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im1 = cv2.resize(im1_orig, None, None, fx=im_scale, fy=im_scale,
                     interpolation=cv2.INTER_LINEAR)
    im2 = cv2.resize(im2_orig, None, None, fx=im_scale, fy=im_scale,
                       interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)

    im = [im1, im2]

    processed_ims.append(im)

  assert len(im_scale_factors) == 1, "Single batch only"

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)


def _get_blobs(im):
  """Convert an image and RoIs within that image into network inputs."""
  im_blob, im_scale_factors = _get_image_blob(im)

  blobs = {'data': im_blob[0], 'data_lwir': im_blob[1]}

  return blobs, im_scale_factors


def im_detect_demo(sess, net, im):
  blobs, im_scales = _get_blobs(im)
  assert len(im_scales) == 1, "Only single-image batch implemented"

  im_blob = blobs['data']
  blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)

  rois, _, _, _, _, cls_score = net.test_image(sess, blobs)

  rpn_boxes = rois[:, 1:5] / im_scales[0]
  bcn_scores = cls_score

  return rpn_boxes, bcn_scores
