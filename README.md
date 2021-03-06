### Multispectral Pedestrian Detection via Simultaneous Detection and Segmentation
Edited by Chengyang Li, Zhejiang University.

Demo code of our paper [Multispectral Pedestrian Detection via Simultaneous Detection and Segmentation](https://arxiv.org/abs/1808.04818) by Chengyang Li, Dan Song, Ruofeng Tong and Min Tang. BMVC 2018. [[project link]](https://li-chengyang.github.io/home/MSDS-RCNN/).

<img src="figures/overview.png" width="800px" height="400px"/>

### Demo
0. Prerequisites

　Basic Tensorflow and Python package installation.
  
　This code is tested on [Ubuntu14.04, tf1.2, Python2.7] and [Ubuntu16.04, tf1.11, Python3.5].

1. Clone the repository
  ```Shell
  git clone https://github.com/Li-Chengyang/MSDS-RCNN.git
  ```

2. Update your -arch in setup script to match your GPU
  ```Shell
  cd MSDS-RCNN/lib
  # Change the GPU architecture (-arch) if necessary
  vim setup.py
  ```

3. Build the Cython modules
  ```Shell
  make clean
  make
  cd ..
  ```

4. Download the pre-trained model

　VGG16 model [[OneDrive]](https://zjueducn-my.sharepoint.com/:u:/g/personal/licy_cs_zju_edu_cn/EU2AEq-VibhPvTvavkM77C8B8o-vG8akiGhlexwNQ8g7fQ?e=nGO5F9) trained on KAIST using original training annotaions.

　VGG16 model [[OneDrive]](https://zjueducn-my.sharepoint.com/:u:/g/personal/licy_cs_zju_edu_cn/EUTtXDnU3vpBoM_0xyGNrRQB1BxFmcwbnIrY3ZHnIYYErw?e=6bbOYC) trained on KAIST using sanitized training annotaions.
  ```Shell
  # Untar files to output/
  cd output
  tar -xvf pretrained.tar
  tar -xvf pretrained_sanitized.tar
  ```
  
5. Run demo

　Model pre-trained on the orignial training annotations
  ```Shell
  python tools/demo.py
  ```

　Model pre-trained on the sanitized training annotations
  ```Shell
  python tools/demo.py --dataset sanitized
  ```

### Detection performance

<img src="figures/comparisons.png" width="800px" height="250px"/>

**Note**: 
Since the original annotations of the test set contain many problematic bounding boxes, we use the [improved testing annotations](http://paul.rutgers.edu/%7Ejl1322/multispectral.htm) provided by Liu et al. to enable a reliable comparison.

### Acknowledgements

Thanks to Xinlei Chen, this pipeline is largely built on his example tensorflow Faster R-CNN code available at:
[https://github.com/endernewton/tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn)

### Citing our paper
If you find our work useful in your research, please consider citing:

```
@InProceedings{li_2018_BMVC,
  author = {Li, Chengyang and Song, Dan and Tong, Ruofeng and Tang, Min},
  title = {Multispectral Pedestrian Detection via Simultaneous Detection and Segmentation},
  booktitle = {British Machine Vision Conference (BMVC)},
  month = {September}
  year = {2018}
}
```

