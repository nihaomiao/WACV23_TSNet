# testing demo for dace dataset
import sys
sys.path.append("/home/hfn5052/code/WACV23_TSNet")
import argparse
import torch
from torch.utils import data
import numpy as np
import cv2
import torch.backends.cudnn as cudnn
import os
import time
from PIL import Image
from utils.misc import Logger
from model.TSNet_pose import TSNet
from dataset.dataset_video_pose import PoseDatasetTestVideo
from utils.misc import vl2ch, vl2im
import sys
from copy import deepcopy
import random
import imageio
from tqdm import tqdm


BATCH_SIZE = 1
TYPE = "pose"
IMG_MEAN = np.array((101.84807705937696, 112.10832843463207, 111.65973036298041), dtype=np.float32)
root_dir = '/data/hfn5052/ResVideoGen/wacv/demo-pose'  # change to your working directory
GPU = "3"
N_BLOCKS = 4
N_DOWNSAMPLING = 3
TRAIN_N_SOURCE = 3
TEST_N_SOURCE = 3
MAX_FRAME_NUM = 30
postfix_pre = "-FS-PS%d-MS%d-D%d-mask-pose" % (N_BLOCKS, TRAIN_N_SOURCE, N_DOWNSAMPLING)
RESTORE_FROM = "/data/hfn5052/VideoGen/BranchGAN_pose/snapshots" \
               + postfix_pre + "/BranchGAN_B0010_S063000.pth"  # change to the path to pretrained models
NUM_EPOCH = 900
if not os.path.exists(RESTORE_FROM):
    print("not existing trained model!")
    exit(-1)
postfix = postfix_pre + "-S%02d" % (TEST_N_SOURCE)

sub_json_pth = "/home/hfn5052/code/WACV23_TSNet/dataset/json_pose/clean_video_dict.json"
msk_json_pth = "/home/hfn5052/code/WACV23_TSNet/dataset/json_pose/clean_unseen_video_dict.json"
label_dir_pth = "/home/hfn5052/code/WACV23_TSNet/demo/dance_example/labels"
image_dir_pth = "/home/hfn5052/code/WACV23_TSNet/demo/dance_example/images"
test_pairs = ["110 164"]
# for how to generate smooth pose labels from original openpose output,
# please refer to dataset/smooth_pose_keypoint.py
smooth_label_path = "/home/hfn5052/code/WACV23_TSNet/dataset/json_pose/smooth_openpose"

basic_point_only = False
remove_face_labels = False

INPUT_SIZE = '256, 256'
RANDOM_SEED = 1234
CKPT_DIR = os.path.join(root_dir, 'ckpt_' + format(NUM_EPOCH, "03d") + postfix)
os.makedirs(CKPT_DIR, exist_ok=True)
IMG_DIR = os.path.join(CKPT_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)
VID_DIR = os.path.join(CKPT_DIR, "videos")
os.makedirs(VID_DIR, exist_ok=True)
LOG_PATH = os.path.join(CKPT_DIR, format(NUM_EPOCH, "03d") + postfix + ".log")
sys.stdout = Logger(LOG_PATH, sys.stdout)
print(postfix)
print("RESTORE_FROM:", RESTORE_FROM)
print("num epoch:", NUM_EPOCH)
print("max frame num of target video", MAX_FRAME_NUM)
print("max frame num of source video", MAX_FRAME_NUM)
print("test pairs:", test_pairs)


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="TS-Net")
    parser.add_argument("--num-workers", default=1)
    parser.add_argument("--gpu", default=GPU,
                        help="choose gpu device.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument('--print-freq', '-p', default=1, type=int,
                        metavar='N', help='print frequency')
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--restore-from", default=RESTORE_FROM)
    return parser.parse_args()


args = get_arguments()


def sample_img(rec_img_batch):
    rec_img = rec_img_batch.data.cpu().numpy()
    img_mean = (torch.from_numpy(IMG_MEAN).cuda() / 255).data.cpu().numpy()
    rec_img = rec_img.transpose(1, 2, 0)
    rec_img = rec_img + img_mean
    rec_img[rec_img < 0] = 0
    rec_img[rec_img > 1] = 1
    rec_img *= 255
    rec_img = cv2.cvtColor(rec_img, cv2.COLOR_BGR2RGB)
    return rec_img


def main():
    """Create the model and start the training."""

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cudnn.enabled = True
    cudnn.benchmark = True
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)

    model = TSNet(is_train=False, label_nc=25,
                  n_blocks=N_BLOCKS,
                  n_downsampling=N_DOWNSAMPLING,
                  n_source=TEST_N_SOURCE,
                  use_mask=True, mean=IMG_MEAN)

    if os.path.isfile(args.restore_from):
        print("=> loading checkpoint '{}'".format(args.restore_from))
        checkpoint = torch.load(args.restore_from)
        model.img_enc.load_state_dict(checkpoint['img_enc'])
        model.lbl_enc.load_state_dict(checkpoint['lbl_enc'])
        model.fuse_net.load_state_dict(checkpoint['fuse_net'])
        model.dec.load_state_dict(checkpoint['dec'])
        print("=> loaded checkpoint '{}'".format(args.restore_from))
    else:
        print("=> no checkpoint found at '{}'".format(args.restore_from))
        exit(-1)

    model.eval()

    testloader = data.DataLoader(PoseDatasetTestVideo(test_pairs=test_pairs,
                                                      sub_json_path=sub_json_pth,
                                                      msk_json_path=msk_json_pth,
                                                      label_path=label_dir_pth,
                                                      image_path=image_dir_pth,
                                                      mean=IMG_MEAN,
                                                      n_frame_total=MAX_FRAME_NUM,
                                                      basic_point_only=basic_point_only,
                                                      remove_face_labels=remove_face_labels,
                                                      smooth_label_path=smooth_label_path),
                                 batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers)

    batch_time = AverageMeter()
    msk_size, _ = map(int, args.input_size.split(','))
    with torch.no_grad():
        end = time.time()
        cnt = 0
        for i_iter, batch in enumerate(testloader):
            src_imgs, src_lbls, src_bboxs, src_names, tar_imgs, tar_lbls, tar_bboxs, tar_names = batch

            tar_imgs = tar_imgs.squeeze(dim=0)
            tar_lbls = tar_lbls.squeeze(dim=0)
            tar_bboxs = tar_bboxs.squeeze(dim=0)
            tar_lbls_resize = vl2ch(tar_lbls, TYPE)
            bs = tar_imgs.size(0)

            src_imgs = src_imgs.squeeze(dim=0)
            src_lbls = src_lbls.squeeze(dim=0)
            src_bboxs = src_bboxs.squeeze(dim=0)
            src_lbls_resize = vl2ch(src_lbls, TYPE)
            src_bs = src_imgs.size(0)

            src_vid_name = src_names[0][0].split('_')[0]
            tar_vid_name = tar_names[0][0].split('_')[0]

            cur_ref_ind = random.sample(list(range(src_bs)), TEST_N_SOURCE)

            ref_imgs = deepcopy(src_imgs[cur_ref_ind, :, :, :])  # n_source, c, h, w
            ref_lbls_resize = deepcopy(src_lbls_resize[cur_ref_ind, :, :, :])
            ref_bboxs = deepcopy(src_bboxs[cur_ref_ind, :, :])

            ref_img_list = ref_imgs.unsqueeze(dim=1)
            ref_bbox_list = ref_bboxs.unsqueeze(dim=1)
            ref_lbl_resize_list = ref_lbls_resize.unsqueeze(dim=1)

            renorm_ref_img = ref_img_list[0] / 255.0
            ref_mean = renorm_ref_img.view(1, 3, -1).mean(dim=2).view(1, 3, 1, 1)
            ref_std = renorm_ref_img.view(1, 3, -1).std(dim=2).view(1, 3, 1, 1)

            new_im_list = []
            for ind in tqdm(range(bs)):
                model.set_test_input(src_img_list=ref_img_list,
                                     src_lbl_list=ref_lbl_resize_list,
                                     src_bbox_list=ref_bbox_list,
                                     tar_lbl=tar_lbls_resize[ind].unsqueeze(dim=0),
                                     tar_bbox=tar_bboxs[ind].unsqueeze(dim=0))
                model.forward()

                rec_tar_imgs = model.rec_tar_img.data.cpu()
                gen_mean = rec_tar_imgs.view(1, 3, -1).mean(dim=2).view(1, 3, 1, 1)
                gen_std = rec_tar_imgs.view(1, 3, -1).std(dim=2).view(1, 3, 1, 1)
                norm_rec_tar_imgs = (rec_tar_imgs - gen_mean)/gen_std
                rec_tar_imgs = norm_rec_tar_imgs*ref_std + ref_mean
                rec_tar_img = sample_img(rec_tar_imgs[0])

                if ind < src_bs:
                    src_img = src_imgs.data.cpu().numpy().copy()[ind]
                    src_img = src_img.transpose(1, 2, 0)
                    src_img += IMG_MEAN
                else:
                    src_img = np.zeros(shape=src_img.shape, dtype=np.float32)
                src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)

                if ind < src_bs:
                    src_lbl = src_lbls.data.cpu().numpy().copy()[ind]
                    src_lbl = vl2im(src_lbl, TYPE)
                else:
                    src_lbl = np.zeros(shape=src_lbl.shape, dtype=np.uint8)

                tar_img = tar_imgs.data.cpu().numpy().copy()[ind]
                tar_img = tar_img.transpose(1, 2, 0)
                tar_img += IMG_MEAN
                tar_img = cv2.cvtColor(tar_img, cv2.COLOR_BGR2RGB)

                # save result image
                new_deb_im = Image.new('RGB', (msk_size * 3, msk_size))
                new_deb_im.paste(Image.fromarray(src_img.astype('uint8'), "RGB"), (0, 0))
                new_deb_im.paste(Image.fromarray(tar_img.astype('uint8'), "RGB"), (msk_size, 0))
                new_deb_im.paste(Image.fromarray(rec_tar_img.astype('uint8'), "RGB"), (msk_size * 2, 0))
                new_deb_im_name = format(cnt, "06d") + "_" + src_vid_name + "_" + tar_names[ind][0]
                dir_name = src_vid_name + "_" + tar_vid_name
                dir_pth = os.path.join(IMG_DIR, dir_name)
                if not os.path.exists(dir_pth):
                    os.makedirs(dir_pth)
                new_deb_im_file = os.path.join(dir_pth, new_deb_im_name)
                new_deb_im.save(new_deb_im_file+".png")
                new_im_list.append(np.asarray(new_deb_im))
                cnt += 1

            # save videos
            video_pth = os.path.join(VID_DIR, dir_name+".gif")
            imageio.mimsave(video_pth, new_im_list)

            batch_time.update(time.time() - end)
            end = time.time()

    print('The total test time is ' + str(batch_time.sum))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
