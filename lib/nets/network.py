# -------------------------------------------------------------------------
# MSDS R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Chengyang Li, based on code from Ross Girshick and Xinlei Chen
# -------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope

import numpy as np

from layer_utils.snippets import generate_anchors_pre
from layer_utils.proposal_layer import proposal_layer, pad_rois
from layer_utils.proposal_layer_combine import proposal_layer_combine_rpn, proposal_layer_combine_bcn
from layer_utils.fusion_layer import fusion_layer
from utils.visualization import draw_bounding_boxes
from model.config import cfg

class Network(object):
  def __init__(self):
    self._predictions = {}
    self._losses = {}
    self._anchor_targets = {}
    self._proposal_targets = {}
    self._layers = {}
    self._gt_image = None
    self._act_summaries = []
    self._score_summaries = {}
    self._train_summaries = []
    self._event_summaries = {}
    self._variables_to_fix = {}
    self._variables_to_assign = {}

  def _add_gt_image(self):
    # add back mean
    image = self._image + cfg.PIXEL_MEANS
    # BGR to RGB (opencv uses BGR)
    resized = tf.image.resize_bilinear(image, tf.to_int32(self._im_info[:2] / self._im_info[2]))
    self._gt_image = tf.reverse(resized, axis=[-1])

  def _add_gt_image_summary(self):
    # use a customized visualization function to visualize the boxes
    if self._gt_image is None:
      self._add_gt_image()
    image = tf.py_func(draw_bounding_boxes, 
                      [self._gt_image, self._gt_boxes, self._im_info],
                      tf.float32, name="gt_boxes")
    
    return tf.summary.image('GROUND_TRUTH', image)

  def _add_act_summary(self, tensor):
    tf.summary.histogram('ACT/' + tensor.op.name + '/activations', tensor)
    tf.summary.scalar('ACT/' + tensor.op.name + '/zero_fraction',
                      tf.nn.zero_fraction(tensor))

  def _add_score_summary(self, key, tensor):
    tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)

  def _add_train_summary(self, var):
    tf.summary.histogram('TRAIN/' + var.op.name, var)

  def _reshape_layer(self, bottom, num_dim, name):
    input_shape = tf.shape(bottom)
    with tf.variable_scope(name) as scope:
      # change the channel to the caffe format
      to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
      # then force it to have channel 2
      reshaped = tf.reshape(to_caffe,
                            tf.concat(axis=0, values=[[1, num_dim, -1], [input_shape[2]]]))
      # then swap the channel back
      to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
      return to_tf

  def _softmax_layer(self, bottom, name):
    if name.startswith('rpn_cls_prob_reshape'):
      input_shape = tf.shape(bottom)
      bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
      reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
      return tf.reshape(reshaped_score, input_shape)
    return tf.nn.softmax(bottom, name=name)

  def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, mode, name):
    with tf.variable_scope(name) as scope:
      rois, roi_scores, rpn_scores = tf.py_func(proposal_layer,
                                                [rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, self._im_info, mode,
                                                 self._anchors, self._num_anchors],
                                                [tf.float32, tf.float32, tf.float32], name="proposal")
      rois.set_shape([None, 5])
      roi_scores.set_shape([None, 1])
      rpn_scores.set_shape([None, 2])
    return rois, roi_scores, rpn_scores

  def _rois_pad_layer(self, rois, is_training, name):
    with tf.variable_scope(name) as scope:
      rois_pad = tf.py_func(pad_rois,
                            [rois, self._im_info, is_training],
                            tf.float32, name='pad')
      rois_pad.set_shape([None, 5])

    return rois_pad

  # Only use it if you have roi_pooling op written in tf.image
  def _roi_pool_layer(self, bootom, rois, name):
    with tf.variable_scope(name) as scope:
      return tf.image.roi_pooling(bootom, rois,
                                  pooled_height=cfg.POOLING_SIZE,
                                  pooled_width=cfg.POOLING_SIZE,
                                  spatial_scale=1. / 16.)[0]

  def _crop_pool_layer(self, bottom, rois, name):
    with tf.variable_scope(name) as scope:
      batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
      # Get the normalized coordinates of bounding boxes
      bottom_shape = tf.shape(bottom)
      height = tf.to_float(bottom_shape[1]) - 1.
      width = tf.to_float(bottom_shape[2]) - 1.
      x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
      y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
      x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
      y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
      # Won't be back-propagated to rois anyway, but to save time
      bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
      crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [cfg.BCN_INPUT_H, cfg.BCN_INPUT_W], name="crops")

    return crops

  def _dropout_layer(self, bottom, name, ratio=0.5):
    return tf.nn.dropout(bottom, ratio, name=name)

  def _anchor_component(self):
    with tf.variable_scope('ANCHOR_' + self._tag) as scope:
      # just to get the shape right
      height = tf.to_int32(tf.ceil(self._im_info[0] / np.float32(self._feat_stride[0])))
      width = tf.to_int32(tf.ceil(self._im_info[1] / np.float32(self._feat_stride[0])))
      anchors, anchor_length = tf.py_func(generate_anchors_pre,
                                          [height, width,
                                           self._feat_stride, self._anchor_scales, self._anchor_ratios],
                                          [tf.float32, tf.int32], name="generate_anchors")
      anchors.set_shape([None, 4])
      anchor_length.set_shape([])
      self._anchors = anchors
      self._anchor_length = anchor_length

  def _build_network(self, is_training, initializer, initializer_bbox):
    """
    RGB + thermal modalities input
    fusion after block3
    """
    # body network
    net_rgb, net_lwir, net_multi = self._image_to_rpn(is_training and cfg.STAGE_TYPE == 'rpn', initializer)

    with tf.variable_scope(self._scope, self._scope):
      # build the anchors for the image
      self._anchor_component()
      # region proposal network
      self._region_proposal(net_rgb, is_training, initializer, suffix='_rgb')
      self._region_proposal(net_lwir, is_training, initializer, suffix='_lwir')
      self._region_proposal(net_multi, is_training, initializer, suffix='_multi')
      if cfg.USE_SEMANTIC_RPN:
        self._seg_predict(net_rgb, is_training and cfg.STAGE_TYPE == 'rpn', initializer, suffix='_rpn_rgb')
        self._seg_predict(net_lwir, is_training and cfg.STAGE_TYPE == 'rpn', initializer, suffix='_rpn_lwir')
        self._seg_predict(net_multi, is_training and cfg.STAGE_TYPE == 'rpn', initializer, suffix='_rpn_multi')

      # combine props
      if (not is_training) and cfg.STAGE_TYPE == 'rpn':
        rois, roi_scores = self._region_proposal_combine_rpn()
      if cfg.STAGE_TYPE == 'bcn':
        rois, roi_scores = self._region_proposal_combine_bcn()

    if cfg.STAGE_TYPE == 'bcn':
      # pad rois with eg.20% in all sides
      rois_pad = self._rois_pad_layer(rois, is_training, 'rois_pad')

      pool_rgb = self._crop_pool_layer(self._image, rois_pad, 'pool_rgb')
      pool_lwir = self._crop_pool_layer(self._lwir, rois_pad, 'pool_lwir')

      bcn_rgb, bcn_lwir, bcn_multi = self._image_to_bcn(pool_rgb, pool_lwir, is_training, initializer)

      fc7_rgb = self._bcn_to_tail(bcn_rgb, is_training, initializer, suffix='_rgb')
      fc7_lwir = self._bcn_to_tail(bcn_lwir, is_training, initializer, suffix='_lwir')
      fc7_multi = self._bcn_to_tail(bcn_multi, is_training, initializer, suffix='_multi')

      self._region_classification(fc7_rgb, is_training, initializer, initializer_bbox, suffix='_rgb')
      self._region_classification(fc7_lwir, is_training, initializer, initializer_bbox, suffix='_lwir')
      self._region_classification(fc7_multi, is_training, initializer, initializer_bbox, suffix='_multi')

      if cfg.USE_SEMANTIC_BCN:
        self._seg_predict(bcn_rgb, is_training, initializer, suffix='_rgb')
        self._seg_predict(bcn_lwir, is_training, initializer, suffix='_lwir')
        self._seg_predict(bcn_multi, is_training, initializer, suffix='_multi')

      # testing stage
      if not is_training:
        self._region_classification_combine()

    self._score_summaries.update(self._predictions)

  def _seg_predict(self, net_seg, is_training, initializer, suffix=''):
    segmap = slim.conv2d(net_seg, 2, [1, 1], trainable=is_training, weights_initializer=initializer,
                         scope='seg_conv'+suffix)

    seg_prob = tf.nn.softmax(segmap, name='seg_prob')

    self._predictions['seg' + suffix] = segmap
    self._predictions['seg_prob' + suffix] = seg_prob

  def _region_proposal(self, net_conv, is_training, initializer, suffix=''):
    rpn = slim.conv2d(net_conv, cfg.RPN_CHANNELS, [3, 3], trainable=is_training and cfg.STAGE_TYPE == 'rpn',
                      weights_initializer=initializer, scope='rpn_conv'+suffix)
    self._act_summaries.append(rpn)
    rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training and cfg.STAGE_TYPE == 'rpn',
                                weights_initializer=initializer,
                                padding='VALID', activation_fn=None, scope='rpn_cls_score'+suffix)
    # change it so that the score has 2 as its channel size
    rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape'+suffix)
    rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape"+suffix)
    rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape, [-1, 2]), axis=1, name="rpn_cls_pred"+suffix)
    rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob"+suffix)
    rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training and cfg.STAGE_TYPE == 'rpn',
                                weights_initializer=initializer,
                                padding='VALID', activation_fn=None, scope='rpn_bbox_pred'+suffix)
    rois, roi_scores, rpn_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, 'TEST', 'rois'+suffix)

    self._predictions["rpn_cls_score_reshape"+suffix] = rpn_cls_score_reshape
    self._predictions["rpn_cls_prob"+suffix] = rpn_cls_prob
    self._predictions["rpn_cls_pred"+suffix] = rpn_cls_pred
    self._predictions["rpn_bbox_pred"+suffix] = rpn_bbox_pred
    self._predictions["rois"+suffix] = rois
    self._predictions["roi_scores" + suffix] = roi_scores
    self._predictions["rpn_scores" + suffix] = rpn_scores

  def _region_proposal_combine_bcn(self, name='rois_combine'):
    rois_rgb = self._predictions["rois_rgb"]
    roi_scores_rgb = self._predictions["roi_scores_rgb"]
    rpn_scores_rgb = self._predictions["rpn_scores_rgb"]
    rois_lwir = self._predictions["rois_lwir"]
    roi_scores_lwir = self._predictions["roi_scores_lwir"]
    rpn_scores_lwir = self._predictions["rpn_scores_lwir"]
    rois_multi = self._predictions["rois_multi"]
    roi_scores_multi = self._predictions["roi_scores_multi"]
    rpn_scores_multi = self._predictions["rpn_scores_multi"]

    if cfg.BCN_PROPOSAL == 'combo':
      rois = tf.concat(axis=0, values=[rois_rgb, rois_lwir, rois_multi])
      roi_scores = tf.concat(axis=0, values=[roi_scores_rgb, roi_scores_lwir, roi_scores_multi])
      rpn_scores = tf.concat(axis=0, values=[rpn_scores_rgb, rpn_scores_lwir, rpn_scores_multi])
    elif cfg.BCN_PROPOSAL == 'multi':
      rois = rois_multi
      roi_scores = roi_scores_multi
      rpn_scores = rpn_scores_multi

    with tf.variable_scope(name) as scope:
      rois, roi_scores, rpn_scores = tf.py_func(proposal_layer_combine_bcn,
                                                [rois, roi_scores, rpn_scores, self._mode],
                                                [tf.float32, tf.float32, tf.float32], name="proposal_filter")
      rois.set_shape([None, 5])
      roi_scores.set_shape([None, 1])
      rpn_scores.set_shape([None, self._num_classes])

    self._predictions["rois"] = rois
    self._predictions["roi_scores"] = roi_scores
    self._predictions["rpn_scores"] = rpn_scores

    return rois, roi_scores

  def _region_proposal_combine_rpn(self, name='rois_combine'):
    rois_rgb = self._predictions["rois_rgb"]
    roi_scores_rgb = self._predictions["roi_scores_rgb"]
    rois_lwir = self._predictions["rois_lwir"]
    roi_scores_lwir = self._predictions["roi_scores_lwir"]
    rois_multi = self._predictions["rois_multi"]
    roi_scores_multi = self._predictions["roi_scores_multi"]

    rois_combo3 = tf.concat(axis=0, values=[rois_rgb, rois_lwir, rois_multi])
    roi_scores_combo3 = tf.concat(axis=0, values=[roi_scores_rgb, roi_scores_lwir, roi_scores_multi])

    with tf.variable_scope(name) as scope:
      rois_rgb, roi_scores_rgb \
        = tf.py_func(proposal_layer_combine_rpn,
                     [rois_rgb, roi_scores_rgb, self._mode],
                     [tf.float32, tf.float32], name="proposal_filter_rgb")
      rois_rgb.set_shape([None, 5])
      roi_scores_rgb.set_shape([None, 1])

      rois_lwir, roi_scores_lwir \
        = tf.py_func(proposal_layer_combine_rpn,
                     [rois_lwir, roi_scores_lwir, self._mode],
                     [tf.float32, tf.float32], name="proposal_filter_lwir")
      rois_lwir.set_shape([None, 5])
      roi_scores_lwir.set_shape([None, 1])

      rois_multi, roi_scores_multi \
        = tf.py_func(proposal_layer_combine_rpn,
                     [rois_multi, roi_scores_multi, self._mode],
                     [tf.float32, tf.float32], name="proposal_filter_multi")
      rois_multi.set_shape([None, 5])
      roi_scores_multi.set_shape([None, 1])

      rois_combo3, roi_scores_combo3 \
        = tf.py_func(proposal_layer_combine_rpn,
                     [rois_combo3, roi_scores_combo3, self._mode],
                     [tf.float32, tf.float32], name="proposal_filter_combo3")
      rois_combo3.set_shape([None, 5])
      roi_scores_combo3.set_shape([None, 1])

    self._predictions["rois_rgb"] = rois_rgb
    self._predictions["roi_scores_rgb"] = roi_scores_rgb
    self._predictions["rois_lwir"] = rois_lwir
    self._predictions["roi_scores_lwir"] = roi_scores_lwir
    self._predictions["rois_multi"] = rois_multi
    self._predictions["roi_scores_multi"] = roi_scores_multi
    self._predictions["rois_combo3"] = rois_combo3
    self._predictions["roi_scores_combo3"] = roi_scores_combo3

    if cfg.BCN_PROPOSAL == 'combo':
      rois = rois_combo3
      roi_scores = roi_scores_combo3
    elif cfg.BCN_PROPOSAL == 'multi':
      rois = rois_multi
      roi_scores = roi_scores_multi
    else:
      raise NotImplementedError

    self._predictions["rois"] = rois
    self._predictions["roi_scores"] = roi_scores

    return rois, roi_scores

  def _region_classification(self, fc7, is_training, initializer, initializer_bbox, suffix=''):
    cls_score = slim.fully_connected(fc7, self._num_classes, 
                                       weights_initializer=initializer,
                                       trainable=is_training,
                                       activation_fn=None, scope='cls_score'+suffix)
    cls_prob = self._softmax_layer(cls_score, "cls_prob"+suffix)
    cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred"+suffix)

    self._predictions["cls_score"+suffix] = cls_score
    self._predictions["cls_pred"+suffix] = cls_pred
    self._predictions["cls_prob"+suffix] = cls_prob

  def _region_classification_combine(self, name='fusion'):
    assert(cfg.STAGE_TYPE == 'bcn')

    rpn_scores = self._predictions['rpn_scores']
    cls_score_rgb = self._predictions['cls_score_rgb']
    cls_score_lwir = self._predictions['cls_score_lwir']
    cls_score_multi = self._predictions['cls_score_multi']

    with tf.variable_scope(name) as scope:
      cls_score, cls_score_bcn, cls_score_multi \
          = tf.py_func(fusion_layer,
                       [rpn_scores, cls_score_rgb, cls_score_lwir, cls_score_multi],
                       [tf.float32, tf.float32, tf.float32], name='score_fusion')
      cls_score.set_shape([None, 1])
      cls_score_bcn.set_shape([None, 1])
      cls_score_multi.set_shape([None, 1])

    self._predictions['cls_score'] = cls_score
    self._predictions['cls_score_combo3'] = cls_score_bcn
    self._predictions['cls_score_combomulti'] = cls_score_multi

  def create_architecture_demo(self, mode, num_classes, tag=None,
                          anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    assert mode == 'TEST', 'only for demo'

    self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
    self._lwir = tf.placeholder(tf.float32, shape=[1, None, None, 3])
    self._im_info = tf.placeholder(tf.float32, shape=[3])
    self._tag = tag

    self._num_classes = num_classes
    self._mode = mode
    self._anchor_scales = anchor_scales
    self._num_scales = len(anchor_scales)

    self._anchor_ratios = anchor_ratios
    self._num_ratios = len(anchor_ratios)

    self._num_anchors = self._num_scales * self._num_ratios

    training = mode == 'TRAIN'
    testing = mode == 'TEST'

    assert tag != None

    # handle most of the regularizers here
    weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
    if cfg.TRAIN.BIAS_DECAY:
      biases_regularizer = weights_regularizer
    else:
      biases_regularizer = tf.no_regularizer

    # select initializers
    if cfg.TRAIN.TRUNCATED:
      initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
    else:
      initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

    # list as many types of layers as possible, even if they are not used now
    with arg_scope([slim.conv2d, slim.conv2d_in_plane,
                    slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected], 
                    weights_regularizer=weights_regularizer,
                    biases_regularizer=biases_regularizer, 
                    biases_initializer=tf.constant_initializer(0.0)):
      self._build_network(training, initializer, initializer_bbox)

    layers_to_output = {}

    return layers_to_output

  def get_variables_to_restore(self, variables, var_keep_dic):
    raise NotImplementedError

  def fix_variables(self, sess, pretrained_model):
    raise NotImplementedError

  # only useful during testing mode
  def test_image(self, sess, image):
    feed_dict = {self._image: image['data'],
                 self._lwir: image['data_lwir'],
                 self._im_info: image['im_info']}

    if cfg.STAGE_TYPE == 'rpn':
      rois_rgb, roi_scores_rgb, \
      rois_lwir, roi_scores_lwir, \
      rois_multi, roi_scores_multi, \
      rois, roi_scores = sess.run([self._predictions['rois_rgb'],
                                   self._predictions['roi_scores_rgb'],
                                   self._predictions['rois_lwir'],
                                   self._predictions['roi_scores_lwir'],
                                   self._predictions['rois_multi'],
                                   self._predictions['roi_scores_multi'],
                                   self._predictions['rois'],
                                   self._predictions['roi_scores']],
                                  feed_dict=feed_dict)
      return rois_rgb, roi_scores_rgb, \
             rois_lwir, roi_scores_lwir, \
             rois_multi, roi_scores_multi, \
             rois, roi_scores
    elif cfg.STAGE_TYPE == 'bcn':
      rois, roi_scores, \
      cls_score_rgb, \
      cls_score_lwir, \
      cls_score_multi, \
      cls_score = sess.run([self._predictions['rois'],
                            self._predictions['roi_scores'],
                            self._predictions['cls_score_rgb'],
                            self._predictions['cls_score_lwir'],
                            self._predictions['cls_score_multi'],
                            self._predictions['cls_score']],
                           feed_dict=feed_dict)
      return rois, roi_scores, \
             cls_score_rgb, \
             cls_score_lwir, \
             cls_score_multi, \
             cls_score
    else:
      raise NotImplementedError
