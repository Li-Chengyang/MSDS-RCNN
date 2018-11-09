# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Blob helper functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.config import cfg
import numpy as np
import cv2


def im_list_to_blob(ims):
  """Convert a list of images into a network input.

  Assumes images are already prepared (means subtracted, BGR order, ...).
  """
  max_shape = np.array([im[0].shape for im in ims]).max(axis=0)
  num_images = len(ims)
  blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                  dtype=np.float32)
  blob2 = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                   dtype=np.float32)
  if cfg.USE_SEMANTIC_RPN or cfg.USE_SEMANTIC_BCN:
    blob_seg = np.zeros((num_images, max_shape[0], max_shape[1], 1),
                     dtype=np.float32)
  for i in range(num_images):
    im = ims[i][0]
    blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    im = ims[i][1]
    blob2[i, 0:im.shape[0], 0:im.shape[1], :] = im
    if len(ims[i]) == 3:
      im = ims[i][-1]
      blob_seg[i, 0:im.shape[0], 0:im.shape[1], 0] = im

    if cfg.USE_SEMANTIC_RPN or cfg.USE_SEMANTIC_BCN:
      blob = [blob, blob2, blob_seg]
    else:
      blob = [blob, blob2]

  return blob


def prep_im_for_blob(im, pixel_means, target_size, max_size):
  """Mean subtract and scale an image for use in a blob."""
  im1 = im[0].astype(np.float32, copy=False)
  im1 -= pixel_means
  im_shape = im1.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])
  im_scale = float(target_size) / float(im_size_min)
  # Prevent the biggest axis from being more than MAX_SIZE
  if np.round(im_scale * im_size_max) > max_size:
    im_scale = float(max_size) / float(im_size_max)
  im1 = cv2.resize(im1, None, None, fx=im_scale, fy=im_scale,
                  interpolation=cv2.INTER_LINEAR)

  im2 = im[1].astype(np.float32, copy=False)
  im2 -= pixel_means
  im2 = cv2.resize(im2, None, None, fx=im_scale, fy=im_scale,
                     interpolation=cv2.INTER_LINEAR)

  if cfg.USE_SEMANTIC_RPN or cfg.USE_SEMANTIC_BCN:
    im_seg = im[-1].astype(np.float32, copy=False)
    im_seg = cv2.resize(im_seg, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)

  if cfg.USE_SEMANTIC_RPN or cfg.USE_SEMANTIC_BCN:
    im = [im1, im2, im_seg]
  else:
    im = [im1, im2]


  return im, im_scale
