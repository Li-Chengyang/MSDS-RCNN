# -------------------------------------------------------------------------
# MSDS R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Chengyang Li, based on code from Ross Girshick and Xinlei Chen
# -------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from model.config import cfg
from model.nms_wrapper import nms


def proposal_layer_combine_bcn(proposals, scores, cls_scores, cfg_key):
  """Combine RPN proposals for BCN input
  """
  #print(proposals.shape, scores.shape, cls_scores.shape)
  if type(cfg_key) == bytes:
    cfg_key = cfg_key.decode('utf-8')
  if cfg_key == 'TEST':
    nms_thresh = cfg[cfg_key].RPN_NMS_THRESH
    post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N

    order = scores.ravel().argsort()[::-1]
    proposals = proposals[order, :]
    scores = scores[order]
    cls_scores = cls_scores[order, :]
    # Non-maximal suppression
    keep = nms(np.hstack((proposals[:, 1:], scores)), nms_thresh)
    # Pick th top region proposals after NMS
    if post_nms_topN < len(keep):
      keep = keep[:post_nms_topN]
    proposals = proposals[keep, :]
    scores = scores[keep]
    cls_scores = cls_scores[keep, :]
  elif cfg_key == 'TRAIN':
    nms_thresh = cfg[cfg_key].RPN_NMS_THRESH

    order = scores.ravel().argsort()[::-1]
    proposals = proposals[order, :]
    scores = scores[order]
    cls_scores = cls_scores[order, :]

    # Non-maximal suppression
    keep = nms(np.hstack((proposals[:, 1:], scores)), nms_thresh)
    proposals = proposals[keep, :]
    scores = scores[keep]
    cls_scores = cls_scores[keep, :]
  else:
    raise NotImplementedError

  if cfg.VERBOSE:
    print('PROPOSAL layer. proposals:', scores.size)
  return proposals, scores, cls_scores

def proposal_layer_combine_rpn(proposals, scores, cfg_key):
  """Only for evluation on RPN stage
  """
  assert (cfg_key == 'TEST')
  if type(cfg_key) == bytes:
    cfg_key = cfg_key.decode('utf-8')
  nms_thresh = cfg[cfg_key].RPN_NMS_THRESH
  post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N

  order = scores.ravel().argsort()[::-1]
  proposals = proposals[order, :]
  scores = scores[order]
  # Non-maximal suppression
  keep = nms(np.hstack((proposals[:, 1:], scores)), nms_thresh)
  # Pick th top region proposals after NMS
  if post_nms_topN < len(keep):
    keep = keep[:post_nms_topN]
  proposals = proposals[keep, :]
  scores = scores[keep]

  if cfg.VERBOSE:
    print('PROPOSAL layer. proposals:', scores.size)
  return proposals, scores
