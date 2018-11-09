# -------------------------------------------------------------------------
# MSDS R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Chengyang Li, based on code from Ross Girshick, Jingjing Liu and Xinlei Chen
# -------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datasets.imdb import imdb
import numpy as np
import scipy.sparse
import pickle
import subprocess
from model.config import cfg


class kaist(imdb):
  def __init__(self, image_set, modality, data_filter, imp_type, devkit_path=None):
    imdb.__init__(self, 'kaist_' + image_set + '_' + modality + '_' + data_filter + imp_type)
    self._image_set = image_set
    self._modality = modality
    if modality == 'multi':
      print('set cfg.MULTI_INPUT -> True')
      cfg.MULTI_INPUT = True
    self._data_filter = data_filter
    self._imp_type = imp_type
    self._data_path = self._get_default_path()
    self._classes = ('__background__',  # always index 0
                     'person')
    self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
    self._image_ext = '.png'
    self._image_index = self._load_image_set_index()
    # Default to roidb handler
    self._roidb_handler = self.gt_roidb

    assert os.path.exists(self._data_path), \
      'Kaist path does not exist: {}'.format(self._data_path)

  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self.image_path_from_index(self._image_index[i])

  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    visible_path = os.path.join(self._data_path, self._image_set, 'images', 'visible',
                              index + self._image_ext)
    lwir_path = os.path.join(self._data_path, self._image_set, 'images', 'lwir',
                              index + self._image_ext)
    seg_path = os.path.join(self._data_path, self._image_set, 'segs' + self._imp_type,
                              index + self._image_ext)
    if self._modality == 'visible':
      if cfg.USE_SEMANTIC_RPN or cfg.USE_SEMANTIC_BCN:
        image_path = [visible_path, seg_path]
      else:
        image_path = [visible_path]
    elif self._modality == 'lwir':
      if cfg.USE_SEMANTIC_RPN or cfg.USE_SEMANTIC_BCN:
        image_path = [lwir_path, seg_path]
      else:
        image_path = [lwir_path]
    elif self._modality == 'multi':
      if cfg.USE_SEMANTIC_RPN or cfg.USE_SEMANTIC_BCN:
        image_path = [visible_path, lwir_path, seg_path]
      else:
        image_path = [visible_path, lwir_path]
    else:
      raise NotImplementedError
    for pth in image_path[:-1]:
      assert os.path.exists(pth), \
        'Path does not exist: {}'.format(pth)
    return image_path

  def _load_image_set_index(self):
    """
    Load the indexes listed in this dataset's image set file.
    """
    # Example path to image set file:
    # self._data_path + /imageSets/train.txt
    image_set_file = os.path.join(self._data_path, 'imageSets',
                                  self._image_set + self._imp_type + '.txt')
    assert os.path.exists(image_set_file), \
      'Path does not exist: {}'.format(image_set_file)
    with open(image_set_file) as f:
      image_index = [x.strip() for x in f.readlines()]
    return image_index

  def _get_default_path(self):
    """
    Return the default path where PASCAL VOC is expected to be installed.
    """
    return os.path.join(cfg.DATA_DIR, 'kaist')

  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.

    This function loads/saves from/to a cache file to speed up future calls.
    """
    cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
    if os.path.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        try:
          roidb = pickle.load(fid)
        except:
          roidb = pickle.load(fid, encoding='bytes')
      print('{} gt roidb loaded from {}'.format(self.name, cache_file))
      return roidb

    gt_roidb = [self._load_kaist_annotation(index)
                for index in self.image_index]
    with open(cache_file, 'wb') as fid:
      pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))

    return gt_roidb

  def _load_kaist_annotation(self, index):
    """
    Load image and bounding boxes info from txt file in the caltech format.
    """
    filename = os.path.join(self._data_path, self._image_set, 'annotations_' + self._data_filter + self._imp_type, index + '.txt')

    with open(filename) as f:
      lines = f.readlines()

    num_pers = 0
    num_igns = 0
    for obj in lines:
      info = obj.split()
      if info[0] == 'person':
        num_pers += 1
      elif info[0] == 'ignore':
        num_igns += 1
      else:
        raise NotImplementedError

    boxes = np.zeros((num_pers, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_pers), dtype=np.int32)
    overlaps = np.zeros((num_pers, self.num_classes), dtype=np.float32)
    # "Seg" area for pascal is just the box area
    seg_areas = np.zeros((num_pers), dtype=np.float32)

    ign_boxes = np.zeros((num_igns, 4), dtype=np.uint16)

    # Load object bounding boxes into a data frame.
    ixp = 0
    ixi = 0
    for obj in lines:
      # Make pixel indexes 0-based
      info = obj.split()
      x1 = float(info[1]) - 1
      y1 = float(info[2]) - 1
      x2 = float(info[3]) - 1
      y2 = float(info[4]) - 1
      if info[0] == 'person':
        cls = self._class_to_ind[info[0]]
        boxes[ixp, :] = [x1, y1, x2, y2]
        gt_classes[ixp] = cls
        overlaps[ixp, cls] = 1.0
        seg_areas[ixp] = (x2 - x1 + 1) * (y2 - y1 + 1)
        ixp = ixp + 1
      elif info[0] == 'ignore':
        ign_boxes[ixi, :] = [x1, y1, x2, y2]
        ixi = ixi + 1

    overlaps = scipy.sparse.csr_matrix(overlaps)

    return {'boxes': boxes,
            'ign_boxes': ign_boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas}

  def append_flipped_images(self):
    num_images = self.num_images
    widths = self._get_widths()
    for i in range(num_images):
      # person bbs
      boxes = self.roidb[i]['boxes'].copy()
      oldx1 = boxes[:, 0].copy()
      oldx2 = boxes[:, 2].copy()
      boxes[:, 0] = widths[i] - oldx2 - 1
      boxes[:, 2] = widths[i] - oldx1 - 1
      assert (boxes[:, 2] >= boxes[:, 0]).all()
      # ignore bbs
      ign_boxes = self.roidb[i]['ign_boxes'].copy()
      oldx1 = ign_boxes[:, 0].copy()
      oldx2 = ign_boxes[:, 2].copy()
      ign_boxes[:, 0] = widths[i] - oldx2 - 1
      ign_boxes[:, 2] = widths[i] - oldx1 - 1
      assert (ign_boxes[:, 2] >= ign_boxes[:, 0]).all()

      entry = {'boxes': boxes,
               'ign_boxes': ign_boxes,
               'gt_overlaps': self.roidb[i]['gt_overlaps'],
               'gt_classes': self.roidb[i]['gt_classes'],
               'flipped': True}
      self.roidb.append(entry)
    self._image_index = self._image_index * 2

  def evaluate_detections(self, all_boxes, output_dir, suffix=''):
    self._write_kaist_results_file(all_boxes, output_dir, suffix)
    self._do_matlab_eval(output_dir, suffix)

  def _write_kaist_results_file(self, all_boxes, output_dir, suffix=''):
    for cls_ind, cls in enumerate(self.classes):
      if cls == '__background__':
        continue
      detdir = os.path.join(output_dir, 'det'+suffix)
      if not os.path.exists(detdir):
        os.makedirs(detdir)
      for im_ind, index in enumerate(self.image_index):
        filename = os.path.join(
          detdir, index + '.txt'
        )
        with open(filename, 'wt') as f:
          dets = all_boxes[cls_ind][im_ind]
          if dets == []:
            continue
          for k in range(dets.shape[0]):
            f.write('{:s} {:.4f} {:.4f} {:.4f} {:.4f} {:.8f}\n'.
                    format(cls, dets[k, 0],
                           dets[k, 1], dets[k, 2],
                           dets[k, 3], dets[k, 4]))
    print('Writing kaist pedestrian detection results file done.')

  def _do_matlab_eval(self, output_dir='output', suffix=''):
    print('-----------------------------------------------------')
    print('Computing results with the official MATLAB eval code.')
    print('-----------------------------------------------------')
    path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                        'KAISTdevkit-matlab-wrapper')
    cmd = 'cd {} && '.format(path)
    cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
    cmd += '-r "dbstop if error; '
    cmd += 'kaist_eval_full(\'{:s}\',\'{:s}\'); quit;"' \
      .format(os.path.join(output_dir, 'det'+suffix), self._data_path)
    print(('Running:\n{}'.format(cmd)))
    status = subprocess.call(cmd, shell=True)

if __name__ == '__main__':
  from datasets.kaist import kaist

  d = kaist('train', 'rgb')
  res = d.roidb
  from IPython import embed;

  embed()
