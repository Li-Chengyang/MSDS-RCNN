# -------------------------------------------------------------------------
# MSDS R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Chengyang Li, based on code from Ross Girshick and Xinlei Chen
# -------------------------------------------------------------------------
"""
Demo script showing detections in sample multispectral images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect_demo
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16

CLASSES = ('__background__',
           'person')

NETS = {'vgg16': 'vgg16_msds_rcnn_final.ckpt'}
DATASETS= {'original': 'pretrained',
           'sanitized': 'pretrained_sanitized'}

def vis_detections(im, dets, thresh=0.5):
  """Draw detected bounding boxes."""
  inds = np.where(dets[:, -1] >= thresh)[0]
  if len(inds) == 0:
    return

  im1 = im[0]
  im2 = im[1]
  h, _, _ = im1.shape
  im1 = im1[:, :, (2, 1, 0)]
  im2 = im2[:, :, (2, 1, 0)]
  im = np.vstack((im1, im2))

  fig, ax = plt.subplots(figsize=(12, 24))
  ax.imshow(im, aspect='equal')
  for i in inds:
    bbox = dets[i, :4]
    score = dets[i, -1]

    # color
    ax.add_patch(
        plt.Rectangle((bbox[0], bbox[1]),
                      bbox[2] - bbox[0],
                      bbox[3] - bbox[1], fill=False,
                      edgecolor='red', linewidth=3.5)
    )
    ax.text(bbox[0], bbox[1] - 2,
            '{:.3f}'.format(score),
            bbox=dict(facecolor='blue', alpha=0.5),
            fontsize=18, color='white')

    # lwir
    ax.add_patch(
        plt.Rectangle((bbox[0], bbox[1] + h),
                      bbox[2] - bbox[0],
                      bbox[3] - bbox[1], fill=False,
                      edgecolor='red', linewidth=3.5)
    )
    ax.text(bbox[0], bbox[1] - 2 + h,
            '{:.3f}'.format(score),
            bbox=dict(facecolor='blue', alpha=0.5),
            fontsize=18, color='white')

  ax.set_title(('pedestrian detections with '
                'p(box) >= {:.1f}').format(thresh),
               fontsize=25)
  plt.axis('off')
  #plt.tight_layout()
  plt.draw()

def demo(sess, net, image_name):
  """Detect pedestrians in an image using pre-computed model."""

  # Load the demo image
  im1_file = os.path.join(cfg.DATA_DIR, 'demo', image_name + '_visible.png')
  im1 = cv2.imread(im1_file)
  im2_file = os.path.join(cfg.DATA_DIR, 'demo', image_name + '_lwir.png')
  im2 = cv2.imread(im2_file)
  im = [im1, im2]

  # Detect all object classes and regress object bounds
  timer = Timer()
  timer.tic()
  boxes, scores = im_detect_demo(sess, net, im)
  timer.toc()
  print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

  # Visualize detections for each class
  CONF_THRESH = 0.5
  NMS_THRESH = 0.3

  dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
  keep = nms(dets, NMS_THRESH)
  dets = dets[keep, :]
  vis_detections(im, dets, thresh=CONF_THRESH)

def parse_args():
  """Parse input arguments."""
  parser = argparse.ArgumentParser(description='MSDS-RCNN demo')
  parser.add_argument('--net', dest='demo_net', help='Currently only support vgg16',
                      choices=NETS.keys(), default='vgg16')
  parser.add_argument('--dataset', dest='dataset', help='Trained dataset',
                      choices=DATASETS.keys(), default='original')
  args = parser.parse_args()

  return args

if __name__ == '__main__':
  args = parse_args()

  # model path
  demonet = args.demo_net
  dataset = args.dataset
  tfmodel = os.path.join('output', DATASETS[dataset], NETS[demonet])

  if not os.path.isfile(tfmodel + '.meta'):
    raise IOError('{:s} not found.\n Please download the pre-trained model and place it properly.'
                  .format(tfmodel + '.meta'))

  # set config
  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  tfconfig.gpu_options.allow_growth=True

  # init session
  sess = tf.Session(config=tfconfig)
  # load network
  if demonet == 'vgg16':
    net = vgg16()
  else:
    raise NotImplementedError

  # load model
  net.create_architecture_demo("TEST", len(CLASSES), tag='default',
                               anchor_scales=cfg.ANCHOR_SCALES,
                               anchor_ratios=cfg.ANCHOR_RATIOS)
  saver = tf.train.Saver()
  saver.restore(sess, tfmodel)

  print('Loaded network {:s}'.format(tfmodel))

  im_names = ['set06_V001_I00459', 'set08_V001_I02579',
              'set08_V002_I01419', 'set09_V000_I01139',
              'set09_V000_I01959', 'set10_V001_I03279',
              'set11_V000_I00159', 'set11_V000_I01279']

  for im_name in im_names:
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Demo for data/demo/{}'.format(im_name))
    demo(sess, net, im_name)

  plt.show()
