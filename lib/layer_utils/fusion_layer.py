# -------------------------------------------------------------------------
# MSDS R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Chengyang Li
# -------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from model.config import cfg

def fusion_layer(rpn_scores, cls_score_rgb, cls_score_lwir, cls_score_multi):
  """Fusion for detection scores from different modalities / stages
  """
  s0 = rpn_scores[:, 0] + cls_score_rgb[:, 0] + cls_score_lwir[:, 0] + cls_score_multi[:, 0]
  s1 = rpn_scores[:, 1] + cls_score_rgb[:, 1] + cls_score_lwir[:, 1] + cls_score_multi[:, 1]
  scores = np.exp(s1) / (np.exp(s0) + np.exp(s1))
  s30 = cls_score_rgb[:, 0] + cls_score_lwir[:, 0] + cls_score_multi[:, 0]
  s31 = cls_score_rgb[:, 1] + cls_score_lwir[:, 1] + cls_score_multi[:, 1]
  scores_bcn = np.exp(s31) / (np.exp(s30) + np.exp(s31))
  s50 = rpn_scores[:, 0] + cls_score_multi[:, 0]
  s51 = rpn_scores[:, 1] + cls_score_multi[:, 1]
  scores_multi = np.exp(s51) / (np.exp(s50) + np.exp(s51))

  return scores, scores_bcn, scores_multi
