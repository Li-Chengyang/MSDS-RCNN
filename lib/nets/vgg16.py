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

from nets.network import Network
from model.config import cfg

class vgg16(Network):
  def __init__(self):
    Network.__init__(self)
    if cfg.REMOVE_POOLING:
      self._feat_stride = [8, ]
    else:
      self._feat_stride = [16, ]
    self._feat_compress = [1. / float(self._feat_stride[0]), ]
    self._scope = 'vgg_16'

  def _image_to_rpn(self, is_training, initializer, reuse=False):
    """
    RGB + thermal modalities input
    halfway fusion
    """
    with tf.variable_scope(self._scope, self._scope, reuse=reuse):
      net_rgb = slim.repeat(self._image, 2, slim.conv2d, 64, [3, 3],
                            trainable=is_training and 1 > cfg.VGG16.FIXED_BLOCKS, scope='conv1')
      net_rgb = slim.max_pool2d(net_rgb, [2, 2], padding='SAME', scope='pool1')
      net_rgb = slim.repeat(net_rgb, 2, slim.conv2d, 128, [3, 3],
                            trainable=is_training and 2 > cfg.VGG16.FIXED_BLOCKS, scope='conv2')
      net_rgb = slim.max_pool2d(net_rgb, [2, 2], padding='SAME', scope='pool2')
      net_rgb3 = slim.repeat(net_rgb, 3, slim.conv2d, 256, [3, 3],
                             trainable=is_training and 3 > cfg.VGG16.FIXED_BLOCKS, scope='conv3')
      net_rgb = slim.max_pool2d(net_rgb3, [2, 2], padding='SAME', scope='pool3')
      net_rgb = slim.repeat(net_rgb, 3, slim.conv2d, 512, [3, 3],
                            trainable=is_training and 4 > cfg.VGG16.FIXED_BLOCKS, scope='conv4')
      if not cfg.REMOVE_POOLING:
        net_rgb = slim.max_pool2d(net_rgb, [2, 2], padding='SAME', scope='pool4')
      net_rgb = slim.repeat(net_rgb, 3, slim.conv2d, 512, [3, 3],
                            trainable=is_training, scope='conv5')

      net_lwir = slim.repeat(self._lwir, 2, slim.conv2d, 64, [3, 3],
                             trainable=is_training and 1 > cfg.VGG16.FIXED_BLOCKS, scope='conv1_lwir')
      net_lwir = slim.max_pool2d(net_lwir, [2, 2], padding='SAME', scope='pool1_lwir')
      net_lwir = slim.repeat(net_lwir, 2, slim.conv2d, 128, [3, 3],
                             trainable=is_training and 2 > cfg.VGG16.FIXED_BLOCKS, scope='conv2_lwir')
      net_lwir = slim.max_pool2d(net_lwir, [2, 2], padding='SAME', scope='pool2_lwir')
      net_lwir3 = slim.repeat(net_lwir, 3, slim.conv2d, 256, [3, 3],
                              trainable=is_training and 3 > cfg.VGG16.FIXED_BLOCKS, scope='conv3_lwir')
      net_lwir = slim.max_pool2d(net_lwir3, [2, 2], padding='SAME', scope='pool3_lwir')
      net_lwir = slim.repeat(net_lwir, 3, slim.conv2d, 512, [3, 3],
                             trainable=is_training and 4 > cfg.VGG16.FIXED_BLOCKS, scope='conv4_lwir')
      if not cfg.REMOVE_POOLING:
        net_lwir = slim.max_pool2d(net_lwir, [2, 2], padding='SAME', scope='pool4_lwir')
      net_lwir = slim.repeat(net_lwir, 3, slim.conv2d, 512, [3, 3],
                             trainable=is_training, scope='conv5_lwir')

      net_cat = tf.concat(axis=3, values=[net_rgb3, net_lwir3])
      net_multi = slim.conv2d(net_cat, 256, [1, 1], weights_initializer=initializer,
                              trainable=is_training, scope='conv3_multi_redim')
      net_multi = slim.max_pool2d(net_multi, [2, 2], padding='SAME', scope='pool3_multi')
      net_multi = slim.repeat(net_multi, 3, slim.conv2d, 512, [3, 3],
                              trainable=is_training and 4 > cfg.VGG16.FIXED_BLOCKS, scope='conv4_multi')
      if not cfg.REMOVE_POOLING:
        net_multi = slim.max_pool2d(net_multi, [2, 2], padding='SAME', scope='pool4_multi')
      net_multi = slim.repeat(net_multi, 3, slim.conv2d, 512, [3, 3],
                              trainable=is_training, scope='conv5_multi')

    self._act_summaries.append(net_multi)
    self._layers['head'] = net_multi

    return net_rgb, net_lwir, net_multi

  def _image_to_rpn_single(self, is_training, initializer, reuse=False):
    """
    Single modality input
    """
    with tf.variable_scope(self._scope, self._scope, reuse=reuse):
      net_rgb = slim.repeat(self._image, 2, slim.conv2d, 64, [3, 3],
                            trainable=is_training and 1 > cfg.VGG16.FIXED_BLOCKS, scope='conv1')
      self._tensor4debug['net_rgb1'] = net_rgb
      net_rgb = slim.max_pool2d(net_rgb, [2, 2], padding='SAME', scope='pool1')
      net_rgb = slim.repeat(net_rgb, 2, slim.conv2d, 128, [3, 3],
                            trainable=is_training and 2 > cfg.VGG16.FIXED_BLOCKS, scope='conv2')
      net_rgb = slim.max_pool2d(net_rgb, [2, 2], padding='SAME', scope='pool2')
      net_rgb = slim.repeat(net_rgb, 3, slim.conv2d, 256, [3, 3],
                             trainable=is_training and 3 > cfg.VGG16.FIXED_BLOCKS, scope='conv3')
      net_rgb = slim.max_pool2d(net_rgb, [2, 2], padding='SAME', scope='pool3')
      net_rgb = slim.repeat(net_rgb, 3, slim.conv2d, 512, [3, 3],
                            trainable=is_training and 4 > cfg.VGG16.FIXED_BLOCKS, scope='conv4')
      if not cfg.REMOVE_POOLING:
        net_rgb = slim.max_pool2d(net_rgb, [2, 2], padding='SAME', scope='pool4')
      net_rgb = slim.repeat(net_rgb, 3, slim.conv2d, 512, [3, 3],
                            trainable=is_training, scope='conv5')

    self._act_summaries.append(net_rgb)
    self._layers['head'] = net_rgb

    self._tensor4debug['net_rgb'] = net_rgb

    return net_rgb

  def _image_to_bcn(self, pool_rgb, pool_lwir, is_training, initializer, reuse=False):
    """
    RGB + thermal modalities input
    halfway fusion
    """
    with tf.variable_scope(self._scope + '_bcn', self._scope + '_bcn', reuse=reuse):
      net_rgb = slim.repeat(pool_rgb, 2, slim.conv2d, 64, [3, 3],
                        trainable=is_training and 1 > cfg.VGG16.FIXED_BLOCKS, scope='conv1')
      net_rgb = slim.max_pool2d(net_rgb, [2, 2], padding='SAME', scope='pool1')
      net_rgb = slim.repeat(net_rgb, 2, slim.conv2d, 128, [3, 3],
                        trainable=is_training and 2 > cfg.VGG16.FIXED_BLOCKS, scope='conv2')
      net_rgb = slim.max_pool2d(net_rgb, [2, 2], padding='SAME', scope='pool2')
      net_rgb3 = slim.repeat(net_rgb, 3, slim.conv2d, 256, [3, 3],
                        trainable=is_training and 3 > cfg.VGG16.FIXED_BLOCKS, scope='conv3')
      net_rgb = slim.max_pool2d(net_rgb3, [2, 2], padding='SAME', scope='pool3')
      net_rgb = slim.repeat(net_rgb, 3, slim.conv2d, 512, [3, 3],
                        trainable=is_training and 4 > cfg.VGG16.FIXED_BLOCKS, scope='conv4')
      net_rgb = slim.max_pool2d(net_rgb, [2, 2], stride=[2, 1], padding='SAME', scope='pool4')
      net_rgb = slim.repeat(net_rgb, 3, slim.conv2d, 512, [3, 3],
                        trainable=is_training, scope='conv5')

      net_lwir = slim.repeat(pool_lwir, 2, slim.conv2d, 64, [3, 3],
                        trainable=is_training and 1 > cfg.VGG16.FIXED_BLOCKS, scope='conv1_lwir')
      net_lwir = slim.max_pool2d(net_lwir, [2, 2], padding='SAME', scope='pool1_lwir')
      net_lwir = slim.repeat(net_lwir, 2, slim.conv2d, 128, [3, 3],
                        trainable=is_training and 2 > cfg.VGG16.FIXED_BLOCKS, scope='conv2_lwir')
      net_lwir = slim.max_pool2d(net_lwir, [2, 2], padding='SAME', scope='pool2_lwir')
      net_lwir3 = slim.repeat(net_lwir, 3, slim.conv2d, 256, [3, 3],
                        trainable=is_training and 3 > cfg.VGG16.FIXED_BLOCKS, scope='conv3_lwir')
      net_lwir = slim.max_pool2d(net_lwir3, [2, 2], padding='SAME', scope='pool3_lwir')
      net_lwir = slim.repeat(net_lwir, 3, slim.conv2d, 512, [3, 3],
                        trainable=is_training and 4 > cfg.VGG16.FIXED_BLOCKS, scope='conv4_lwir')
      net_lwir = slim.max_pool2d(net_lwir, [2, 2], stride=[2, 1], padding='SAME', scope='pool4_lwir')
      net_lwir = slim.repeat(net_lwir, 3, slim.conv2d, 512, [3, 3],
                        trainable=is_training, scope='conv5_lwir')

      net_cat = tf.concat(axis=3, values=[net_rgb3, net_lwir3])
      net_multi = slim.conv2d(net_cat, 256, [1, 1], weights_initializer=initializer,
                        trainable=is_training, scope='conv3_multi_redim')
      net_multi = slim.max_pool2d(net_multi, [2, 2], padding='SAME', scope='pool3_multi')
      net_multi = slim.repeat(net_multi, 3, slim.conv2d, 512, [3, 3],
                        trainable=is_training and 4 > cfg.VGG16.FIXED_BLOCKS, scope='conv4_multi')
      net_multi = slim.max_pool2d(net_multi, [2, 2], stride=[2, 1], padding='SAME', scope='pool4_multi')
      net_multi = slim.repeat(net_multi, 3, slim.conv2d, 512, [3, 3],
                        trainable=is_training, scope='conv5_multi')

    self._act_summaries.append(net_multi)
    self._layers['head'] = net_multi

    return net_rgb, net_lwir, net_multi

  def _bcn_to_tail(self, pool5, is_training, initializer, suffix='', reuse=False):
    with tf.variable_scope(self._scope + '_bcn', self._scope + '_bcn', reuse=reuse):
      pool5_flat = slim.flatten(pool5, scope='flatten'+suffix)
      fc6 = slim.fully_connected(pool5_flat, 4096, weights_initializer=initializer,
                                 trainable=is_training, scope='fc6'+suffix)
      if is_training:
        fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True, 
                            scope='dropout6'+suffix)
      fc7 = slim.fully_connected(fc6, 4096, weights_initializer=initializer,
                                 trainable=is_training, scope='fc7'+suffix)
      if is_training:
        fc7 = slim.dropout(fc7, keep_prob=0.5, is_training=True, 
                            scope='dropout7'+suffix)

    return fc7

  def get_variables_to_restore(self, variables, var_keep_dic, stage_type):
    variables_to_restore = []

    for v in variables:
      # dicts for pretrained params
      if v.name.split(':')[0] in var_keep_dic:
        self._variables_to_assign[v.name] = v
      # exclude the first conv layer to swap RGB to BGR
      if v.name == (self._scope + '/conv1/conv1_1/weights:0'):
        self._variables_to_fix[v.name] = v
        if stage_type == 'rpn':
          continue
      if v.name.split(':')[0] in var_keep_dic:
        print('Variables restored: %s' % v.name)
        variables_to_restore.append(v)

    return variables_to_restore

  def fix_variables(self, sess, pretrained_model, variables):
    print('Fix VGG16 layers..')
    with tf.variable_scope('Fix_VGG16') as scope:
      with tf.device("/cpu:0"):
        # fix the vgg16 issue from conv weights to fc weights
        # fix RGB to BGR
        conv1_rgb = tf.get_variable("conv1_rgb", [3, 3, 3, 64], trainable=False)

        restorer_fc = tf.train.Saver({self._scope + "/conv1/conv1_1/weights": conv1_rgb})
        restorer_fc.restore(sess, pretrained_model)

        sess.run(tf.assign(self._variables_to_fix[self._scope + '/conv1/conv1_1/weights:0'], tf.reverse(conv1_rgb, [2])))

        # restore other branches
        for v in variables:
          if ('_lwir' in v.name) and self._variables_to_assign.has_key(v.name.replace('_lwir', '')):
            sess.run(tf.assign(v, self._variables_to_assign[v.name.replace('_lwir', '')]))
          if ('_multi' in v.name) and self._variables_to_assign.has_key(v.name.replace('_multi', '')):
            sess.run(tf.assign(v, self._variables_to_assign[v.name.replace('_multi', '')]))
          if cfg.USE_SEMANTIC_RPN:
            if ('_seg' in v.name) and self._variables_to_assign.has_key(v.name.replace('_seg', '')):
              sess.run(tf.assign(v, self._variables_to_assign[v.name.replace('_seg', '')]))

  def fix_variables_bcn(self, sess, variables):
    print('Init bcn layers..')
    with tf.variable_scope('Init_bcn') as scope:
      with tf.device("/cpu:0"):
        # restore bcn
        for v in variables:
          if ('_bcn' in v.name) and self._variables_to_assign.has_key(v.name.replace('_bcn', '')):
            print('Variables initialized:', v.name)
            sess.run(tf.assign(v, self._variables_to_assign[v.name.replace('_bcn', '')]))
