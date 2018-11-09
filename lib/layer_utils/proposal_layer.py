# -------------------------------------------------------------------------
# MSDS R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Chengyang Li, based on code from Ross Girshick and Xinlei Chen
# -------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from model.config import cfg
from model.bbox_transform import bbox_transform_inv, clip_boxes
from model.nms_wrapper import nms

def proposal_layer(rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, im_info, cfg_key, anchors, num_anchors):
  """Proposal layer
  """
  if type(cfg_key) == bytes:
    cfg_key = cfg_key.decode('utf-8')
  is_training = cfg_key == 'TRAIN'
  if is_training:
    pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
    post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
  conf_thresh = cfg[cfg_key].RPN_CONF_THRESH
  nms_thresh = cfg[cfg_key].RPN_NMS_THRESH
  # Get the scores and bounding boxes
  scores = rpn_cls_prob[:, :, :, num_anchors:]
  scores = scores.reshape((-1, 1))
  scores_cls0 = rpn_cls_score[:, :, :, :num_anchors]
  scores_cls1 = rpn_cls_score[:, :, :, num_anchors:]
  scores_cls0 = scores_cls0.reshape((-1, 1))
  scores_cls1 = scores_cls1.reshape((-1, 1))
  scores_cls = np.hstack((scores_cls0, scores_cls1))
  rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))
  proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
  proposals = clip_boxes(proposals, im_info[:2])

  if is_training:
    if cfg.STAGE_TYPE == 'rpn':
      # Pick the top region proposals
      order = scores.ravel().argsort()[::-1]
      if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
      proposals = proposals[order, :]
      scores = scores[order]
      scores_cls = scores_cls[order, :]
      # Non-maximal suppression
      keep = nms(np.hstack((proposals, scores)), nms_thresh)
      # Pick th top region proposals after NMS
      if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
      proposals = proposals[keep, :]
      scores = scores[keep]
      scores_cls = scores_cls[keep, :]
    elif cfg.STAGE_TYPE == 'bcn':
      # use thresh 0.01 to filter most boxes
      keep = np.where(scores > conf_thresh)[0]
      proposals = proposals[keep, :]
      scores = scores[keep]
      scores_cls = scores_cls[keep, :]
    else:
      raise NotImplementedError
  else: # testing
    # use thresh 0.01 to filter most boxes
    keep = np.where(scores > conf_thresh)[0]
    proposals = proposals[keep, :]
    scores = scores[keep]
    scores_cls = scores_cls[keep, :]

  # Only support single image as input
  batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
  blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

  if cfg.VERBOSE:
    print('PROPOSAL layer. proposals:', scores.size)

  return blob, scores, scores_cls

def pad_rois(rois, im_info, is_training):
  """Pad rois to utilize contextual information and to alleviate truncation
  """
  nroi = rois.shape[0]
  proposals = np.zeros((nroi, 4), dtype=np.float32)

  w = rois[:, 3] - rois[:, 1]
  h = rois[:, 4] - rois[:, 2]
  dw = cfg.POOL_PAD_RATIO * w
  dh = cfg.POOL_PAD_RATIO * h

  nroi = rois.shape[0]
  if is_training:
    nw = npr.rand(nroi)
    nh = npr.rand(nroi)
  else:
    nw = np.ones(nroi) * 0.5
    nh = np.ones(nroi) * 0.5

  proposals[:, 0] = rois[:, 1] - (dw - nw * (1 + 2 * cfg.POOL_PAD_RATIO) / 15 * w)
  proposals[:, 1] = rois[:, 2] - (dh - nh * (1 + 2 * cfg.POOL_PAD_RATIO) / 15 * h)
  proposals[:, 2] = rois[:, 3] + (dw - (1-nw) * (1 + 2 * cfg.POOL_PAD_RATIO) / 15 * w)
  proposals[:, 3] = rois[:, 4] + (dh - (1-nh) * (1 + 2 * cfg.POOL_PAD_RATIO) / 15 * h)

  proposals = clip_boxes(proposals, im_info[:2])

  batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
  blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

  return blob
