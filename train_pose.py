# training TS-Net for Youtube-dance dataset

import argparse
import torch
from torch.utils import data
import numpy as np
import cv2
import torch.backends.cudnn as cudnn
import os
import os.path as osp
import timeit
import math
from PIL import Image
from utils.misc import Logger
from model.TSNet_pose import TSNet
from dataset.dataset_video_pose import PoseDatasetTrainVideoMask
from utils.misc import vl2ch, vl2im
import sys
import random

start = timeit.default_timer()
N_BLOCKS = 4
N_DOWNSAMPLING = 3
N_SOURCE = 3
TYPE = "pose"
basic_point_only = False
remove_face_labels = False
label_nc = 25
BATCH_SIZE = 10
N_FRAME_TOTAL = 10
INTERVAL = 4
INITIAL_EPOCH = 400  # the initial epochs using the fixed learning rate, taking starting epochs into consideration
MAX_EPOCH = 900
IMG_MEAN = np.array((101.84807705937696, 112.10832843463207, 111.65973036298041), dtype=np.float32)
root_dir = '/data/TSNet_pose'
json_pth = "/data/youtube-dance/clean_video_dict.json"
label_dir_pth = "/data/youtube-dance/checked_openpose"
image_dir_pth = "/data/youtube-dance/checked_images"
GPU = "5"
postfix = "-PS%d-MS%d-D%d-mask-pose" % (N_BLOCKS, N_SOURCE, N_DOWNSAMPLING)
INPUT_SIZE = '256, 256'
LEARNING_RATE = 2e-4
LAMBDA_FML = 10.0
LAMBDA_VGG = 10.0
LAMBDA_CON = 10.0
LAMBDA_GRAD = 10.0
RANDOM_SEED = 1234
RESTORE_FROM = ""
SNAPSHOT_DIR = os.path.join(root_dir, 'snapshots' + postfix)
IMGSHOT_DIR = os.path.join(root_dir, 'imgshots' + postfix)
NUM_EXAMPLES_PER_EPOCH = 100 * (N_FRAME_TOTAL-N_SOURCE)
NUM_STEPS_PER_EPOCH = math.ceil(NUM_EXAMPLES_PER_EPOCH / float(BATCH_SIZE))
MAX_ITER = max(NUM_EXAMPLES_PER_EPOCH * MAX_EPOCH + 1,
               NUM_STEPS_PER_EPOCH * BATCH_SIZE * MAX_EPOCH + 1)
INITIAL_ITER = NUM_EXAMPLES_PER_EPOCH * INITIAL_EPOCH
POWER = 1.0
SAVE_PRED_EVERY = NUM_STEPS_PER_EPOCH * (MAX_EPOCH // 10)

if not os.path.exists(SNAPSHOT_DIR):
    os.makedirs(SNAPSHOT_DIR)
if not os.path.exists(IMGSHOT_DIR):
    os.makedirs(IMGSHOT_DIR)

LOG_PATH = SNAPSHOT_DIR + "/B" + format(BATCH_SIZE, "04d") + "E" + format(MAX_EPOCH, "04d") + ".log"
sys.stdout = Logger(LOG_PATH, sys.stdout)
print(postfix)
print(json_pth)
print("RESTORE_FROM", RESTORE_FROM)
print("num examples per epoch:", NUM_EXAMPLES_PER_EPOCH)
print("lambda_fml", LAMBDA_FML)
print("lambda_vgg", LAMBDA_VGG)
print("lambda_grad", LAMBDA_GRAD)
print("initial epoch:", INITIAL_EPOCH)
print("max epoch:", MAX_EPOCH)
print("power:", POWER)
print("num frame total, interval:", N_FRAME_TOTAL, INTERVAL)
print("basic_point_only, remove_face_labels", basic_point_only, remove_face_labels)
print("save every:", SAVE_PRED_EVERY)


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="TSNet")
    parser.add_argument("--fine-tune", default=False)
    parser.add_argument("--set-start", default=False)
    parser.add_argument("--start-step", default=0, type=int)
    parser.add_argument("--label-dir-pth", default=label_dir_pth)
    parser.add_argument("--img-dir", type=str, default=IMGSHOT_DIR,
                        help="Where to save images of the model.")
    parser.add_argument("--num-workers", default=10)
    parser.add_argument("--final-step", type=int, default=int(NUM_STEPS_PER_EPOCH * MAX_EPOCH),
                        help="Number of training steps.")
    parser.add_argument("--gpu", default=GPU,
                        help="choose gpu device.")
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency')
    parser.add_argument('--save-img-freq', default=200, type=int,
                        metavar='N', help='save image frequency')
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-list", type=str, default=json_pth,
                        help="Path to the text file listing the images in the dataset.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--random-mirror", default=True,
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-jitter", default=True)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", default=RESTORE_FROM)
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    return parser.parse_args()


args = get_arguments()
print("use mirror, jitter?",
      args.random_mirror, args.random_jitter)


def sample_img(rec_img_batch):
    rec_img = rec_img_batch[0].data.cpu().numpy().copy()
    img_mean = (torch.from_numpy(IMG_MEAN).cuda() / 255).data.cpu().numpy()
    rec_img = rec_img.transpose(1, 2, 0)
    rec_img = rec_img + img_mean
    rec_img[rec_img < 0] = 0
    rec_img[rec_img > 1] = 1
    rec_img *= 255
    return cv2.cvtColor(rec_img, cv2.COLOR_BGR2RGB)


def process_img(img_batch):
    bs, c, h, w = img_batch.size()
    return (img_batch.view(bs, h, w, c) - torch.tensor(IMG_MEAN).cuda()).view(bs, c, h, w)


def main():
    """Create the model and start the training."""

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cudnn.enabled = True
    cudnn.benchmark = True
    setup_seed(args.random_seed)

    model = TSNet(lr=LEARNING_RATE,
                  n_blocks=N_BLOCKS,
                  n_source=N_SOURCE,
                  n_downsampling=N_DOWNSAMPLING,
                  lambda_FML=LAMBDA_FML, lambda_VGG=LAMBDA_VGG, lambda_CON=LAMBDA_CON,
                  lambda_GRAD=LAMBDA_GRAD,
                  is_train=True, getIntermFeat=True, label_nc=label_nc, use_mask=True, mean=IMG_MEAN)

    if args.fine_tune:
        raise NotImplementedError
    elif args.restore_from:
        if os.path.isfile(args.restore_from):
            print("=> loading checkpoint '{}'".format(args.restore_from))
            checkpoint = torch.load(args.restore_from)
            if args.set_start:
                args.start_step = int(math.ceil(checkpoint['example'] / args.batch_size))
            model.img_enc.load_state_dict(checkpoint['img_enc'])
            model.lbl_enc.load_state_dict(checkpoint['lbl_enc'])
            if 'dec' in checkpoint.keys():
                model.dec.load_state_dict(checkpoint['dec'])
            model.fuse_net.load_state_dict(checkpoint["fuse_net"])
            model.netD.load_state_dict(checkpoint['netD'])
            if 'netDF' in checkpoint.keys():
                model.netDF.load_state_dict(checkpoint['netDF'])
            else:
                print("face discriminator is not existing!")
            print("=> loaded checkpoint '{}'".format(args.restore_from))
        else:
            print("=> no checkpoint found at '{}'".format(args.restore_from))
    else:
        print("NO checkpoint found!")

    model.train()

    print("the architecture of image encoder:")
    print(model.img_enc)
    print("the architecture of label encoder:")
    print(model.lbl_enc)
    print("the architecture of decoder:")
    print(model.dec)
    print("the architecture of fusion net:")
    print(model.fuse_net)
    print("the architecture of discriminator:")
    print(model.netD)
    print("the architecture of face discriminator:")
    print(model.netDF)

    trainloader = data.DataLoader(PoseDatasetTrainVideoMask(label_path=label_dir_pth,
                                                            image_path=image_dir_pth,
                                                            json_path=json_pth,
                                                            n_frame_total=N_FRAME_TOTAL,
                                                            is_mirror=args.random_mirror,
                                                            is_jitter=args.random_jitter,
                                                            mean=IMG_MEAN,
                                                            interval=INTERVAL,
                                                            basic_point_only=basic_point_only,
                                                            remove_face_labels=remove_face_labels),
                                  batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True)

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses_G_GAN = AverageMeter()
    losses_G_FML = AverageMeter()
    losses_G_VGG = AverageMeter()
    losses_GF_GAN = AverageMeter()
    losses_GF_FML = AverageMeter()
    losses_GF_VGG = AverageMeter()
    losses_D_real = AverageMeter()
    losses_D_fake = AverageMeter()
    losses_DF_real = AverageMeter()
    losses_DF_fake = AverageMeter()

    losses_grad = AverageMeter()
    losses_warp = AverageMeter()

    cnt = 0
    actual_step = args.start_step
    model.setup(actual_step=actual_step,
                batch_size=args.batch_size,
                initial_iter=INITIAL_ITER,
                max_iter=MAX_ITER,
                power=POWER)

    while actual_step < args.final_step:
        iter_end = timeit.default_timer()
        # the begin of one epoch
        model.print_learning_rate()
        for i_iter, batch in enumerate(trainloader):
            ori_src_imgs_list, ori_src_lbls_list, ori_src_bboxs_list, ori_src_name_list = batch
            bs = ori_src_imgs_list[0].size(0)
            src_imgs_list = ori_src_imgs_list[:N_SOURCE]
            src_lbls_list = ori_src_lbls_list[:N_SOURCE]
            src_bboxs_list = ori_src_bboxs_list[:N_SOURCE]
            src_name_list = ori_src_name_list[:N_SOURCE]
            use_prev = [False] * N_SOURCE
            for frame_iter in range(N_SOURCE, N_FRAME_TOTAL):
                tar_imgs = ori_src_imgs_list[frame_iter]
                tar_lbls = ori_src_lbls_list[frame_iter]
                tar_bboxs = ori_src_bboxs_list[frame_iter]
                tar_name = ori_src_name_list[frame_iter]
                actual_step = int(args.start_step + cnt)
                data_time.update(timeit.default_timer() - iter_end)

                model.setup(actual_step=actual_step,
                            batch_size=args.batch_size,
                            initial_iter=INITIAL_ITER,
                            max_iter=MAX_ITER,
                            power=POWER)

                src_lbls_resize_list = [vl2ch(src_lbls, TYPE, basic_point_only=basic_point_only,
                                              remove_face_labels=remove_face_labels) for src_lbls in src_lbls_list]
                tar_lbls_resize = vl2ch(tar_lbls, TYPE, basic_point_only=basic_point_only,
                                        remove_face_labels=remove_face_labels)

                model.set_train_input(src_img_list=src_imgs_list,
                                      src_lbl_list=src_lbls_resize_list,
                                      src_bbox_list=src_bboxs_list,
                                      tar_img=tar_imgs,
                                      tar_lbl=tar_lbls_resize,
                                      tar_bbox=tar_bboxs,
                                      use_prev=use_prev)

                model.optimize_parameters()

                batch_time.update(timeit.default_timer() - iter_end)
                iter_end = timeit.default_timer()

                losses_term = model.get_current_losses()
                losses_G_GAN.update(losses_term['G_GAN'], bs)
                losses_G_FML.update(losses_term['G_FML'], bs)
                losses_G_VGG.update(losses_term['G_VGG'], bs)
                losses_GF_GAN.update(losses_term['GF_GAN'], bs)
                losses_GF_FML.update(losses_term['GF_FML'], bs)
                losses_GF_VGG.update(losses_term['GF_VGG'], bs)
                losses_D_real.update(losses_term['D_real'], bs)
                losses_D_fake.update(losses_term['D_fake'], bs)
                losses_DF_real.update(losses_term['DF_real'], bs)
                losses_DF_fake.update(losses_term['DF_fake'], bs)

                losses_grad.update(losses_term['grad_G'], bs)
                losses_warp.update(losses_term['warp'], bs)

                if actual_step % args.print_freq == 0:
                    print('iter: [{0}]{1}/{2}\n'
                          'loss_G_GAN {loss_G_GAN.val:.4f} ({loss_G_GAN.avg:.4f})\t'
                          'loss_G_VGG {loss_G_VGG.val:.4f} ({loss_G_VGG.avg:.4f})\t'
                          'loss_G_FML {loss_G_FML.val:.4f} ({loss_G_FML.avg:.4f})\n'
                          'loss_GF_GAN {loss_GF_GAN.val:.4f} ({loss_GF_GAN.avg:.4f})\t'
                          'loss_GF_VGG {loss_GF_VGG.val:.4f} ({loss_GF_VGG.avg:.4f})\t'
                          'loss_GF_FML {loss_GF_FML.val:.4f} ({loss_GF_FML.avg:.4f})\n'
                          'loss_D_real {loss_D_real.val:.4f} ({loss_D_real.avg:.4f})\t'
                          'loss_D_fake {loss_D_fake.val:.4f} ({loss_D_fake.avg:.4f})\n'
                          'loss_DF_real {loss_DF_real.val:.4f} ({loss_DF_real.avg:.4f})\t'
                          'loss_DF_fake {loss_DF_fake.val:.4f} ({loss_DF_fake.avg:.4f})\n'
                          'loss_grad {loss_grad.val:.4f} ({loss_grad.avg:.4f})\t'
                          'loss_warp {loss_warp.val:.4f} ({loss_warp.avg:.4f})\n'
                        .format(
                        cnt, actual_step, args.final_step,
                        batch_time=batch_time,
                        data_time=data_time,
                        loss_G_GAN=losses_G_GAN,
                        loss_G_VGG=losses_G_VGG,
                        loss_G_FML=losses_G_FML,
                        loss_GF_GAN=losses_GF_GAN,
                        loss_GF_VGG=losses_GF_VGG,
                        loss_GF_FML=losses_GF_FML,
                        loss_D_real=losses_D_real,
                        loss_D_fake=losses_D_fake,
                        loss_DF_real=losses_DF_real,
                        loss_DF_fake=losses_DF_fake,
                        loss_grad=losses_grad,
                        loss_warp=losses_warp
                    ))

                if actual_step % args.save_img_freq == 0:
                    img_name_postfix = ""
                    src_img_list = []
                    for i in range(model.n_source):
                        is_prev = use_prev[i]
                        if is_prev:
                            src_img = sample_img(src_imgs_list[i])
                        else:
                            src_img = src_imgs_list[i].data.cpu().numpy().copy()[0]
                            src_img = src_img.transpose(1, 2, 0)
                            src_img += IMG_MEAN
                            src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
                        src_img_list.append(src_img)
                    tar_img = tar_imgs.data.cpu().numpy().copy()[0]
                    tar_img = tar_img.transpose(1, 2, 0)
                    tar_img += IMG_MEAN
                    tar_img = cv2.cvtColor(tar_img, cv2.COLOR_BGR2RGB)

                    rec_tar_imgs = model.rec_tar_img
                    msk_size = rec_tar_imgs.size(2)
                    rec_tar_img = sample_img(rec_tar_imgs)

                    save_warp_src_img_list = []
                    for i in range(model.n_source):
                        save_warp_src_img = sample_img(model.warp_src_img_list[i])
                        save_warp_src_img_list.append(save_warp_src_img)


                    src_lbl_list = []
                    for i in range(model.n_source):
                        src_lbl = src_lbls_list[i].data.cpu().numpy().copy()[0]
                        src_lbl = vl2im(src_lbl, TYPE,
                                        basic_point_only=basic_point_only,
                                        remove_face_labels=remove_face_labels)
                        src_lbl_list.append(src_lbl)
                    tar_lbl = tar_lbls.data.cpu().numpy().copy()[0]
                    tar_lbl = vl2im(tar_lbl, TYPE,
                                    basic_point_only=basic_point_only,
                                    remove_face_labels=remove_face_labels)

                    tar_face_img = model.crop_face(tar_imgs[0].unsqueeze(0)/255.0, tar_lbls_resize[0].unsqueeze(dim=0))
                    tar_face_img = sample_img(tar_face_img)
                    face_img = model.crop_face(rec_tar_imgs[0].unsqueeze(0), tar_lbls_resize[0].unsqueeze(dim=0))
                    face_img = sample_img(face_img)
                    new_im = Image.new('RGB', (msk_size * (model.n_source + 2), msk_size * 3))
                    for i in range(model.n_source):
                        new_im.paste(Image.fromarray(src_img_list[i].astype('uint8'), 'RGB'), (msk_size * i, 0))
                        new_im.paste(Image.fromarray(src_lbl_list[i].astype('uint8'), 'RGB'), (msk_size * i, msk_size))
                        new_im.paste(Image.fromarray(save_warp_src_img_list[i].astype('uint8'), 'RGB'),
                                     (msk_size*i, msk_size * 2))
                    new_im.paste(Image.fromarray(tar_img.astype('uint8'), 'RGB'), (msk_size * model.n_source, 0))
                    new_im.paste(Image.fromarray(tar_lbl.astype('uint8'), 'RGB'), (msk_size * model.n_source, msk_size))
                    new_im.paste(Image.fromarray(rec_tar_img.astype('uint8'), 'RGB'),
                                 (msk_size * (model.n_source+1), 0))
                    new_im.paste(Image.fromarray(tar_face_img.astype('uint8'), 'RGB').resize((256, 256)),
                                 (msk_size * model.n_source, msk_size * 2))
                    new_im.paste(Image.fromarray(face_img.astype('uint8'), 'RGB').resize((256, 256)),
                                 (msk_size * (model.n_source + 1), msk_size * 2))
                    new_im_name = 'B' + format(args.batch_size, "04d") + '_S' + format(actual_step, "06d") \
                                  + '_' + src_name_list[0][0] + '_to_' + tar_name[0] + img_name_postfix
                    new_im_file = os.path.join(args.img_dir, new_im_name + ".jpg")
                    new_im.save(new_im_file)

                if actual_step % args.save_pred_every == 0: # and cnt != 0:
                    print('taking snapshot ...')
                    torch.save({'example': actual_step * args.batch_size,
                                'img_enc': model.img_enc.state_dict(),
                                'lbl_enc': model.lbl_enc.state_dict(),
                                'dec': model.dec.state_dict(),
                                'fuse_net': model.fuse_net.state_dict(),
                                'netD': model.netD.state_dict(),
                                'netDF': model.netDF.state_dict()},
                               osp.join(args.snapshot_dir,
                                        'TSNet_B' + format(args.batch_size, "04d") + '_S' + format(actual_step,
                                                                                                       "06d") + '.pth'))
                    model.img_enc.cuda()
                    model.lbl_enc.cuda()
                    model.dec.cuda()
                    model.fuse_net.cuda()
                    model.netD.cuda()
                    model.netDF.cuda()

                if actual_step >= args.final_step:
                    break
                cnt += 1

            if actual_step >= args.final_step:
                break

    print('save the final model ...')
    torch.save({'example': actual_step * args.batch_size,
                'img_enc': model.img_enc.state_dict(),
                'lbl_enc': model.lbl_enc.state_dict(),
                'dec': model.dec.state_dict(),
                'fuse_net': model.fuse_net.state_dict(),
                'netD': model.netD.state_dict(),
                'netDF': model.netDF.state_dict()},
               osp.join(args.snapshot_dir,
                        'TSNet_B' + format(args.batch_size, "04d") + '_S' + format(actual_step, "06d") + '.pth'))
    end = timeit.default_timer()
    print(end - start, 'seconds')


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


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()
