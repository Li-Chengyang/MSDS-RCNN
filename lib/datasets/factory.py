# -------------------------------------------------------------------------
# MSDS R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Chengyang Li based on code from Ross Girshick
# -------------------------------------------------------------------------
"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.kaist import kaist

# Set up kaist_<split>
# train  samples every 2 frames
# test   samples every 20 frames
for split in ['train-all02']:
  for modality in ['visible', 'lwir', 'multi']:
    for dfilter in ['50np']:
      for timp in ['', '_or', '_id', '_in', '_ia', '_n1', '_n2', '_n3']:
        name = 'kaist_{}_{}_{}{}'.format(split, modality, dfilter, timp)
        __sets[name] = (lambda split=split, modality=modality, dfilter=dfilter, timp=timp:
                        kaist(split, modality, dfilter, timp))
for split in ['test-all']:
  for modality in ['visible', 'lwir', 'multi']:
    for dfilter in ['50np']:
      for timp in ['']:
        name = 'kaist_{}_{}_{}{}'.format(split, modality, dfilter, timp)
        __sets[name] = (lambda split=split, modality=modality, dfilter=dfilter, timp=timp:
                        kaist(split, modality, dfilter, timp))

def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
