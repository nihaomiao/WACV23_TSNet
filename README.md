TS-Net
====

The pytorch implementation of our WACV23 paper "Cross-identity Video Motion Retargeting with Joint Transformation and Synthesis".

<div align=center><img src="architecture.png" width="787px" height="306px"/></div>

Example videos
----
Some generated video results on FaceForensics dataset.

<div align=center>
<img src="sup-mat/face1.gif" width="250" height="342"/>
<img src="sup-mat/face2.gif" width="250" height="342"/>
</div>

Some generated video results on Youtube-dance dataset.

<div align=center>
<img src="sup-mat/pose1.gif" width="250" height="342"/>
<img src="sup-mat/pose2.gif" width="250" height="342"/>
</div>

Quick Start
----

The following codes show a toy example about how to train TS-Net.
```python
# this code will show a toy example about how to train our model
import torch
from model.TSNet import TSNet
import os

# setting GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# our model requires landmark/keypoint labels
label_nc = 2

bs = 4
# source, i.e., subject videos
# here we make some fake data for illustration
src_img_batch_list = []
src_lbl_batch_list = []
src_bbox_batch_list = []  # bounding box for mask-aware similarity matrix computation
for i in range(3):
    src_img_batch = torch.rand((bs, 3, 256, 256)).cuda()
    src_lbl_batch = torch.randint(low=0, high=2, size=(bs, label_nc, 256, 256)).cuda().to(torch.float32)
    src_bbox_batch = torch.randint(low=0, high=2, size=(bs, 256, 256)).cuda().to(torch.float32)
    src_img_batch_list.append(src_img_batch)
    src_lbl_batch_list.append(src_lbl_batch)
    src_bbox_batch_list.append(src_bbox_batch)

# target, i.e., driving videos
tar_img_batch = torch.rand((bs, 3, 256, 256)).cuda()
tar_lbl_batch = torch.randint(low=0, high=2, size=(bs, label_nc, 256, 256)).cuda().to(torch.float32)
tar_bbox_batch = torch.randint(low=0, high=2, size=(bs, 256, 256)).cuda().to(torch.float32)

# model architecture
model = TSNet(is_train=True, label_nc=label_nc,
              n_blocks=0, debug=False,
              n_downsampling=3,
              n_source=3).cuda()

# setting training input
model.set_train_input(src_img_list=src_img_batch_list,
                      src_lbl_list=src_lbl_batch_list,
                      src_bbox_list=src_bbox_batch_list,
                      tar_img=tar_img_batch, tar_lbl=tar_lbl_batch,
                      tar_bbox=tar_bbox_batch)

# one training step to update TS-Net
model.optimize_parameters()
```

TODO
----
The complete training and testing of TS-Net and upload pretrained models

DONE
----
The architecture of TS-Net

Citing TS-Net
----
TODO

Acknowledgement
----
We borrows some codes from [pix2pixHD](https://github.com/NVIDIA/pix2pixHD), [vid2vid](https://github.com/NVIDIA/vid2vid), and [fs-vid2vid](https://github.com/NVlabs/few-shot-vid2vid).
