import os
import torch.utils.data as data
from PIL import Image, ImageOps
import numpy as np
import random
from utils.keypoint2img import read_keypoints
from utils.keypoint2img_posenorm import read_keypoints_posenorm, read_keypoints_posenorm_smooth, \
    define_edge_lists, read_pts_posenorm
import argparse
import torchvision.transforms.functional as F
from utils.misc import im2vl, check_path_valid
import cv2
import re
import json_tricks as json


# training which will return a bounding box for reducing the computation complexity of attention matrix
class PoseDatasetTrainVideoMask(data.Dataset):
    def __init__(self, json_path, label_path, image_path, mean,
                 n_frame_total, is_jitter, is_mirror,
                 basic_point_only=False,
                 remove_face_labels=False,
                 interval=1):
        super(PoseDatasetTrainVideoMask, self).__init__()
        self.opt = argparse.ArgumentParser()
        #### option setting
        self.opt.basic_point_only = basic_point_only
        self.opt.remove_face_labels = remove_face_labels
        self.opt.isTrain = True
        self.opt.aspect_ratio = 0.5
        ####
        self.interval = interval
        self.t = "pose"
        self.n_frame_total = n_frame_total  # the total number of frames
        self.mean = mean
        self.is_jitter = is_jitter
        self.is_mirror = is_mirror
        self.img_size = (128, 256)
        with open(json_path, "r") as f:
            video_dict = json.load(f)
        name_list = [int(x) for x in list(video_dict.keys())]
        name_list.sort()
        vid_name_list = ["%05d" % x for x in name_list]
        self.lbl_pths = []
        self.img_pths = []
        self.names = []
        for idx, name in enumerate(name_list):
            frame_name_list = video_dict[str(name)]
            frame_name_list.sort()
            frame_list = [os.path.join(image_path, vid_name_list[idx], frame) for frame in frame_name_list]
            self.img_pths.append(frame_list)
            label_list = [os.path.join(label_path, vid_name_list[idx], frame[:-4]+"_keypoints.json") for frame in frame_name_list]
            self.lbl_pths.append(label_list)
            rename_list = [self.rename(frame, vid_name_list[idx]) for frame in frame_name_list]
            self.names.append(rename_list)
        check_path_valid(self.lbl_pths, self.img_pths)

    def __getitem__(self, index):
        # apply random crop/scale/flip
        # if flip, all source frames should flip
        seq_idx = index % len(self.lbl_pths)
        L_paths = self.lbl_pths[seq_idx]
        I_paths = self.img_pths[seq_idx]
        names = self.names[seq_idx]
        # random choose n_frame_total frames from the video
        if len(L_paths) > (self.n_frame_total - 1) * self.interval:
            start_idx = random.choice(list(range(0, len(L_paths) - (self.n_frame_total - 1) * self.interval)))
            interval = self.interval
        else:
            start_idx = random.choice(list(range(0, self.n_frame_total)))
            interval = 1
        # reference frame, i.e. first frame
        anchor_size = self.read_data(I_paths[start_idx % len(I_paths)]).size
        _, anchor_crop_coords, anchor_face_pts, anchor_scale = self.get_image(A_path=L_paths[start_idx % len(L_paths)],
                                                                              size=anchor_size,
                                                                              crop_coords=None, input_type="openpose",
                                                                              ref_face_pts=None, scale=None)
        source_img_list = []
        source_lbl_list = []
        source_bbox_list = []
        for i in range(self.n_frame_total):
            src_size = self.read_data(I_paths[(start_idx+i*interval) % len(I_paths)]).size
            src_lbl, src_crop_coords, src_face_pts, _ = self.get_image(A_path=L_paths[(start_idx+i*interval) % len(L_paths)],
                                                                       size=src_size,
                                                                       crop_coords=anchor_crop_coords,
                                                                       input_type="openpose",
                                                                       ref_face_pts=None,
                                                                       scale=anchor_scale)
            src_bbox = self.get_bbox_image(src_lbl)
            src_img = self.get_image(A_path=I_paths[(start_idx+i*interval) % len(I_paths)], size=src_size,
                                     crop_coords=src_crop_coords,
                                     input_type="img")
            source_img_list.append(src_img)
            source_lbl_list.append(src_lbl)
            source_bbox_list.append(src_bbox)

        # resize to (128, 256)
        # then resize to square size
        src_img_list = []
        src_lbl_list = []
        src_bbox_list = []
        for i in range(self.n_frame_total):
            src_img = source_img_list[i].resize(self.img_size)
            src_lbl = source_lbl_list[i].resize(self.img_size, resample=Image.NEAREST)
            src_bbox = source_bbox_list[i].resize(self.img_size, resample=Image.NEAREST)
            src_img = self.resize_square(src_img)
            src_lbl = self.resize_square(src_lbl)
            src_bbox = self.resize_square(src_bbox)
            src_img_list.append(src_img)
            src_lbl_list.append(src_lbl)
            src_bbox_list.append(src_bbox)

        if self.is_jitter:
            bright = 64. / 255
            contrast = 0.25
            sat = 0.25
            hue = 0.04
            bright_f = random.uniform(max(0, 1 - bright), 1 + bright)
            contrast_f = random.uniform(max(0, 1 - contrast), 1 + contrast)
            sat_f = random.uniform(max(0, 1 - sat), 1 + sat)
            hue_f = random.uniform(-hue, hue)

            src_img_list = [F.adjust_brightness(src_img, bright_f) for src_img in src_img_list]
            src_img_list = [F.adjust_contrast(src_img, contrast_f) for src_img in src_img_list]
            src_img_list = [F.adjust_saturation(src_img, sat_f) for src_img in src_img_list]
            src_img_list = [F.adjust_hue(src_img, hue_f) for src_img in src_img_list]

        if self.is_mirror:
            if random.random() < 0.5:
                src_img_list = [F.hflip(src_img) for src_img in src_img_list]
                src_lbl_list = [F.hflip(src_lbl) for src_lbl in src_lbl_list]
                src_bbox_list = [F.hflip(src_bbox) for src_bbox in src_bbox_list]

        src_lbl_arr_list = [np.asarray(src_lbl, dtype=np.uint8) for src_lbl in src_lbl_list]
        src_lbl_arr_list = [im2vl(src_lbl_arr, self.t,
                                  self.opt.basic_point_only,
                                  self.opt.remove_face_labels) for src_lbl_arr in src_lbl_arr_list]
        src_img_arr_list = [cv2.cvtColor(np.asarray(src_img), cv2.COLOR_RGB2BGR) for src_img in src_img_list]
        src_img_arr_list = [np.asarray(src_img_arr, np.float32) for src_img_arr in src_img_arr_list]

        src_img_arr_list = src_img_arr_list - self.mean
        src_img_arr_list = [src_img_arr.transpose((2, 0, 1)) for src_img_arr in src_img_arr_list]

        src_bbox_arr_list = [np.asarray(src_bbox, dtype=np.uint8) for src_bbox in src_bbox_list]
        src_bbox_arr_list = [np.array(src_bbox != 0, dtype=np.uint8) for src_bbox in src_bbox_arr_list]
        src_bbox_arr_list = [src_bbox_arr.copy() for src_bbox_arr in src_bbox_arr_list]

        src_name_list = []
        for i in range(self.n_frame_total):
            src_name_list.append(names[(start_idx+i*interval) % len(L_paths)])

        return src_img_arr_list, src_lbl_arr_list, src_bbox_arr_list, src_name_list

    def __len__(self):
        return len(self.lbl_pths)

    def rename(self, img_name, vid_name):
        img_idx = int(re.sub("[^0-9]", "", img_name))
        vid_idx = int(re.sub("[^0-9]", "", vid_name))
        return "%03d_frame_%05d" % (vid_idx, img_idx)

    def resize_square(self, img):
        w, h = img.size
        s = max(w, h)
        delta_w = s - w
        delta_h = s - h
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        return ImageOps.expand(img, padding)

    def read_data(self, path, data_type='img'):
        is_img = data_type == 'img'
        if is_img:
            img = Image.open(path)
        elif data_type == 'np':
            img = np.loadtxt(path, delimiter=',')
        else:
            img = path
        return img

    def get_image(self, A_path, size, crop_coords, input_type,
                  ppl_idx=None, ref_face_pts=None, scale=None):
        opt = self.opt
        if input_type == 'openpose':
            # get image from openpose keypoints
            A_img, pose_pts, face_pts = read_keypoints(opt, A_path, size,
                                                       opt.basic_point_only, opt.remove_face_labels,
                                                       ppl_idx,
                                                       ref_face_pts)

            # randomly crop the image
            A_img, crop_coords, scale = self.crop_person_region(A_img, crop_coords, pose_pts, size, scale)

        else:
            A_img = self.read_data(A_path)
            A_img, _, _ = self.crop_person_region(A_img, crop_coords)

        if input_type == 'openpose':
            return A_img, crop_coords, face_pts, scale
        return A_img

    # only crop the person region in the image
    def crop_person_region(self, A_img, crop_coords, pose_pts=None, size=None, scale=None):
        # get the crop coordinates
        if crop_coords is None:
            offset_max = 0.05
            random_offset = [random.uniform(-offset_max, offset_max),
                             random.uniform(-offset_max, offset_max)] if self.opt.isTrain else [0, 0]
            crop_coords, scale = self.get_crop_coords(pose_pts, size, random_offset, scale)

        # only crop the person region
        if type(A_img) == np.ndarray:
            xs, ys, xe, ye = crop_coords
            A_img = Image.fromarray(A_img[ys:ye, xs:xe, :])
        else:
            A_img = A_img.crop(crop_coords)
        return A_img, crop_coords, scale

    ### get the pixel coordinates to crop
    def get_crop_coords(self, pose_pts, size, offset=None, scale=None):
        w, h = size
        valid = pose_pts[:, 0] != 0
        x, y = pose_pts[valid, 0], pose_pts[valid, 1]

        # get the center position and length of the person to crop
        # ylen, xlen: height and width of the person
        # y_cen, x_cen: center of the person
        x_cen = int(x.min() + x.max()) // 2 if x.shape[0] else w // 2
        if y.shape[0]:
            y_min = max(y.min(), min(pose_pts[15, 1], pose_pts[16, 1]))
            y_max = max(pose_pts[11, 1], pose_pts[14, 1])
            if y_max == 0: y_max = y.max()
            y_cen = int(y_min + y_max) // 2
            y_len = y_max - y_min
        else:
            y_cen = y_len = h // 2

        # randomly scale the crop size for augmentation
        # final cropped size = person height * scale
        if scale is None:
            scale = random.uniform(1.4, 1.6) if self.opt.isTrain else 1.5

        # bh, bw: half of height / width of final cropped size
        bh = int(min(h, max(h // 4, y_len * scale))) // 2
        bw = int(bh * self.opt.aspect_ratio)

        # randomly offset the cropped position for augmentation
        if offset is not None:
            x_cen += int(offset[0] * bw)
            y_cen += int(offset[1] * bh)
        x_cen = max(bw, min(w - bw, x_cen))
        y_cen = max(bh, min(h - bh, y_cen))

        return [(x_cen - bw), (y_cen - bh), (x_cen + bw), (y_cen + bh)], scale

    def get_bbox_image(self, labels):
        labels_arr = np.array(labels)
        labels_arr = np.sum(labels_arr != 0, axis=2)
        labels_nz = np.nonzero(labels_arr)
        labels_y = labels_nz[0]
        labels_x = labels_nz[1]
        y_min, y_max = labels_y.min(), labels_y.max()
        x_min, x_max = labels_x.min(), labels_x.max()
        h, w = labels_arr.shape
        im_bbox = np.zeros((h, w), np.uint8)
        y_margin = h//16
        x_margin = w//16
        x_min = int(max(0.0, x_min - x_margin))
        x_max = int(min(w, x_max + x_margin))
        y_min = int(max(0.0, y_min - y_margin))
        y_max = int(min(h, y_max + y_margin))
        im_bbox[y_min:y_max, x_min:x_max] = 255.0
        return Image.fromarray(im_bbox)


class PoseDatasetTestVideo(data.Dataset):
    def __init__(self, test_pairs,
                 sub_json_path, msk_json_path,
                 label_path, smooth_label_path, image_path, mean,
                 n_frame_total, basic_point_only=False,
                 remove_face_labels=False):
        super(PoseDatasetTestVideo, self).__init__()
        self.pose_edge_list = define_edge_lists(basic_point_only)[0]
        self.img_pth = image_path
        self.lbl_pth = label_path
        self.smooth_lbl_pth = smooth_label_path
        self.opt = argparse.ArgumentParser()
        #### option setting
        self.opt.basic_point_only = basic_point_only
        self.opt.remove_face_labels = remove_face_labels
        self.opt.isTrain = False
        self.opt.aspect_ratio = 0.5
        ####
        self.t = "pose"
        self.n_frame_total = n_frame_total  # the total number of frames
        self.mean = mean
        self.img_size = (128, 256)
        with open(sub_json_path, "r") as f:
            sub_video_dict = json.load(f)
        with open(msk_json_path, "r") as f:
            msk_video_dict = json.load(f)
        self.video_dict = {**sub_video_dict, **msk_video_dict}
        self.img_ids = test_pairs

    def __getitem__(self, index):
        ################
        vid_idx1, vid_idx2 = self.img_ids[index].split(" ")
        is_female1 = (int(vid_idx1) - int("091")) <= 0 or (int(vid_idx1) in [147, 151])
        is_female2 = (int(vid_idx2) - int("091")) <= 0 or (int(vid_idx2) in [147, 151])
        if is_female1 == is_female2:
            diff_sex = ""
        if is_female1 and not is_female2:
            diff_sex = "fm"  # female & male
        if not is_female1 and is_female2:
            diff_sex = "mf"  # male & female

        src_frame_list = self.video_dict[vid_idx1][:self.n_frame_total]
        src_img_list = [os.path.join(self.img_pth, "%05d" % int(vid_idx1), frame) for frame in src_frame_list]
        src_lbl_list = [os.path.join(self.lbl_pth, "%05d" % int(vid_idx1), frame[:-4]+"_keypoints.json") for frame in src_frame_list]
        src_name_list = [self.rename(img_name=frame, vid_name="%05d" % int(vid_idx1)) for frame in src_frame_list]
        src_img_len = len(src_img_list)
        tar_frame_list = self.video_dict[vid_idx2][:self.n_frame_total]
        tar_img_list = [os.path.join(self.img_pth, "%05d" % int(vid_idx2), frame) for frame in tar_frame_list]
        tar_lbl_list = [os.path.join(self.lbl_pth, "%05d" % int(vid_idx2), frame[:-4]+"_keypoints.json") for frame in tar_frame_list]
        tar_name_list = [self.rename(img_name=frame, vid_name="%05d" % int(vid_idx2)) for frame in tar_frame_list]
        num_tar_frame = len(tar_img_list)

        # first frame
        anchor_size = self.read_data(src_img_list[0]).size
        _, anchor_crop_coords, anchor_pts, anchor_scale = self.get_image(A_path=src_lbl_list[0],
                                                                         size=anchor_size,
                                                                         crop_coords=None, input_type="openpose",
                                                                         ref_pts_length=None, scale=None)
        # read source image
        source_img_list = []
        source_lbl_list = []
        source_pts_list = []
        source_bbox_list = []
        for i in range(src_img_len):
            src_size = self.read_data(src_img_list[i]).size
            src_lbl, src_crop_coords, src_pts, _ = self.get_image(A_path=src_lbl_list[i],
                                                                  size=src_size,
                                                                  crop_coords=anchor_crop_coords,
                                                                  input_type="openpose",
                                                                  ref_pts_length=None,
                                                                  scale=anchor_scale)
            src_img = self.get_image(A_path=src_img_list[i], size=src_size, crop_coords=src_crop_coords,
                                     input_type="img")
            src_bbox = self.get_bbox_image(src_lbl)
            source_img_list.append(src_img)
            source_lbl_list.append(src_lbl)
            source_pts_list.append(src_pts)
            source_bbox_list.append(src_bbox)

        # read unpaired target frame
        # read smooth mask
        ppl_idx = None
        smooth_tar_lbl_json = os.path.join(self.smooth_lbl_pth, "%05d.json" % int(vid_idx2))
        with open(smooth_tar_lbl_json, "r") as f:
            smooth_tar_keypoints = json.load(f)
        target_img_list = []
        ori_target_lbl_list = []
        ori_target_pts_list = []
        tar_size = self.read_data(tar_img_list[0]).size
        _, anchor_tar_crop_coords, anchor_tar_pts, _ = self.get_smooth_lbl(keypoints=smooth_tar_keypoints,
                                                                           index=0,
                                                                           size=tar_size,
                                                                           crop_coords=None,
                                                                           input_type="openpose",
                                                                           ref_pts_length=None,
                                                                           scale=anchor_scale,
                                                                           ppl_idx=ppl_idx)

        for i in range(min(self.n_frame_total, num_tar_frame)):
            tar_size = self.read_data(tar_img_list[i]).size
            tar_lbl, tar_crop_coords, tar_pts, _ = self.get_smooth_lbl(keypoints=smooth_tar_keypoints,
                                                                       index=i,
                                                                       size=tar_size,
                                                                       crop_coords=anchor_tar_crop_coords,
                                                                       input_type="openpose",
                                                                       ref_pts_length=None,
                                                                       scale=anchor_scale,
                                                                       ppl_idx=ppl_idx)
            tar_img = self.get_image(A_path=tar_img_list[i], size=tar_size, crop_coords=tar_crop_coords,
                                     input_type="img")
            target_img_list.append(tar_img)
            ori_target_lbl_list.append(tar_lbl)
            ori_target_pts_list.append(tar_pts)

        # get cropped pose image and normalize it
        target_lbl_list = []
        target_bbox_list = []
        for i in range(min(self.n_frame_total, num_tar_frame)):
            cropped_tar_size = ori_target_lbl_list[i].size
            tar_lbl = self.get_image(A_path=None,
                                     size=cropped_tar_size,
                                     crop_coords=anchor_tar_crop_coords,
                                     input_type="pts",
                                     ref_pts_length=diff_sex,
                                     scale=None,
                                     ppl_idx=ppl_idx,
                                     pts=ori_target_pts_list[i])
            tar_bbox = self.get_bbox_image(tar_lbl)
            target_lbl_list.append(tar_lbl)
            target_bbox_list.append(tar_bbox)

        # resize to (128, 256)
        # then resize to square size
        src_img_list = []
        src_lbl_list = []
        src_bbox_list = []
        for i in range(src_img_len):
            src_img = source_img_list[i].resize(self.img_size)
            src_lbl = source_lbl_list[i].resize(self.img_size, resample=Image.NEAREST)
            src_bbox = source_bbox_list[i].resize(self.img_size, resample=Image.NEAREST)
            src_img = self.resize_square(src_img)
            src_lbl = self.resize_square(src_lbl)
            src_bbox = self.resize_square(src_bbox)
            src_img_list.append(src_img)
            src_lbl_list.append(src_lbl)
            src_bbox_list.append(src_bbox)

        tar_img_list = []
        tar_lbl_list = []
        tar_bbox_list = []
        for i in range(min(self.n_frame_total, num_tar_frame)):
            tar_img = target_img_list[i].resize(self.img_size)
            tar_lbl = target_lbl_list[i].resize(self.img_size, resample=Image.NEAREST)
            tar_bbox = target_bbox_list[i].resize(self.img_size, resample=Image.NEAREST)
            tar_img = self.resize_square(tar_img)
            tar_lbl = self.resize_square(tar_lbl)
            tar_bbox = self.resize_square(tar_bbox)
            tar_img_list.append(tar_img)
            tar_lbl_list.append(tar_lbl)
            tar_bbox_list.append(tar_bbox)

        src_lbl_arr_list = [np.asarray(src_lbl, dtype=np.uint8) for src_lbl in src_lbl_list]
        src_lbl_arr_list = [im2vl(src_lbl_arr, self.t,
                                  self.opt.basic_point_only,
                                  self.opt.remove_face_labels) for src_lbl_arr in src_lbl_arr_list]
        src_bbox_arr_list = [np.asarray(src_bbox, dtype=np.uint8) for src_bbox in src_bbox_list]
        src_bbox_arr_list = [np.array(src_bbox != 0, dtype=np.uint8) for src_bbox in src_bbox_arr_list]

        tar_lbl_arr_list = [np.asarray(tar_lbl, dtype=np.uint8) for tar_lbl in tar_lbl_list]
        tar_lbl_arr_list = [im2vl(tar_lbl_arr, self.t,
                                  self.opt.basic_point_only,
                                  self.opt.remove_face_labels) for tar_lbl_arr in tar_lbl_arr_list]
        tar_bbox_arr_list = [np.asarray(tar_bbox, dtype=np.uint8) for tar_bbox in tar_bbox_list]
        tar_bbox_arr_list = [np.array(tar_bbox != 0, dtype=np.uint8) for tar_bbox in tar_bbox_arr_list]

        src_img_arr_list = [cv2.cvtColor(np.asarray(src_img), cv2.COLOR_RGB2BGR) for src_img in src_img_list]
        src_img_arr_list = [np.asarray(src_img_arr, np.float32) for src_img_arr in src_img_arr_list]
        tar_img_arr_list = [cv2.cvtColor(np.asarray(tar_img), cv2.COLOR_RGB2BGR) for tar_img in tar_img_list]
        tar_img_arr_list = [np.asarray(tar_img_arr, np.float32) for tar_img_arr in tar_img_arr_list]

        src_img_arr_list = src_img_arr_list - self.mean
        src_img_arr_list = [src_img_arr.transpose((2, 0, 1)) for src_img_arr in src_img_arr_list]
        tar_img_arr_list = tar_img_arr_list - self.mean
        tar_img_arr_list = [tar_img_arr.transpose((2, 0, 1)) for tar_img_arr in tar_img_arr_list]

        return np.stack(src_img_arr_list, axis=0), np.stack(src_lbl_arr_list, axis=0), np.stack(src_bbox_arr_list, axis=0), src_name_list, \
               np.stack(tar_img_arr_list, axis=0), np.stack(tar_lbl_arr_list, axis=0), np.stack(tar_bbox_arr_list, axis=0), tar_name_list

    def __len__(self):
        return len(self.img_ids)

    def rename(self, img_name, vid_name):
        img_idx = int(re.sub("[^0-9]", "", img_name))
        vid_idx = int(re.sub("[^0-9]", "", vid_name))
        return "%03d_frame_%05d" % (vid_idx, img_idx)

    def resize_square(self, img):
        w, h = img.size
        s = max(w, h)
        delta_w = s - w
        delta_h = s - h
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        return ImageOps.expand(img, padding)

    def read_data(self, path, data_type='img'):
        is_img = data_type == 'img'
        if is_img:
            img = Image.open(path)
        elif data_type == 'np':
            img = np.loadtxt(path, delimiter=',')
        else:
            img = path
        return img

    def get_image(self, A_path, size, crop_coords, input_type,
                  ppl_idx=None, ref_pts_length=None, scale=None, pts=None):
        opt = self.opt
        if input_type == 'openpose':
            # get image from openpose keypoints
            # pose_pts is the original one
            A_img, pose_pts, pts = read_keypoints_posenorm(opt, A_path, size,
                                                           opt.basic_point_only, opt.remove_face_labels,
                                                           ppl_idx,
                                                           ref_pts_length)

            # randomly crop the images
            A_img, crop_coords, scale = self.crop_person_region(A_img, crop_coords, pose_pts, size, scale)

        elif input_type == "img":
            A_img = self.read_data(A_path)
            A_img, _, _ = self.crop_person_region(A_img, crop_coords)
        else:  # pts
            A_img, pose_pts, pts = read_pts_posenorm(opt, pts, crop_coords, size,
                                                     opt.basic_point_only, opt.remove_face_labels,
                                                     ref_pts_length)
            A_img = Image.fromarray(A_img)
        if input_type == 'openpose':
            return A_img, crop_coords, pts, scale
        return A_img

    def get_smooth_lbl(self, keypoints, index, size, crop_coords, input_type,
                       ppl_idx=None, ref_pts_length=None, scale=None, pts=None):
        opt = self.opt
        if input_type == 'openpose':
            # get image from openpose keypoints
            # pose_pts is the original one
            A_img, pose_pts, pts = read_keypoints_posenorm_smooth(opt, keypoints, index, size,
                                                                  opt.basic_point_only, opt.remove_face_labels,
                                                                  ppl_idx,
                                                                  ref_pts_length)

            # randomly crop the images
            A_img, crop_coords, scale = self.crop_person_region(A_img, crop_coords, pose_pts, size, scale)
        else:  # pts
            A_img, pose_pts, pts = read_pts_posenorm(opt, pts, crop_coords, size,
                                                     opt.basic_point_only, opt.remove_face_labels,
                                                     ref_pts_length)
            A_img = Image.fromarray(A_img)
        if input_type == 'openpose':
            return A_img, crop_coords, pts, scale
        return A_img

    # only crop the person region in the image
    def crop_person_region(self, A_img, crop_coords, pose_pts=None, size=None, scale=None):
        # get the crop coordinates
        if crop_coords is None:
            offset_max = 0.05
            random_offset = [0, 0]
            crop_coords, scale = self.get_crop_coords(pose_pts, size, random_offset, scale)

        # only crop the person region
        if type(A_img) == np.ndarray:
            xs, ys, xe, ye = crop_coords
            A_img = Image.fromarray(A_img[ys:ye, xs:xe, :])
        else:
            A_img = A_img.crop(crop_coords)
        return A_img, crop_coords, scale

    ### get the pixel coordinates to crop
    def get_crop_coords(self, pose_pts, size, offset=None, scale=None):
        w, h = size
        valid = pose_pts[:, 0] != 0
        x, y = pose_pts[valid, 0], pose_pts[valid, 1]

        # get the center position and length of the person to crop
        # ylen, xlen: height and width of the person
        # y_cen, x_cen: center of the person
        x_cen = int(x.min() + x.max()) // 2 if x.shape[0] else w // 2
        if y.shape[0]:
            y_min = max(y.min(), min(pose_pts[15, 1], pose_pts[16, 1]))
            y_max = max(pose_pts[11, 1], pose_pts[14, 1])
            if y_max == 0: y_max = y.max()
            y_cen = int(y_min + y_max) // 2
            y_len = y_max - y_min
        else:
            y_cen = y_len = h // 2

        # randomly scale the crop size for augmentation
        # final cropped size = person height * scale
        if scale is None:
            scale = 1.5

        # bh, bw: half of height / width of final cropped size
        bh = int(min(h, max(h // 4, y_len * scale))) // 2
        bw = int(bh * self.opt.aspect_ratio)

        # randomly offset the cropped position for augmentation
        if offset is not None:
            x_cen += int(offset[0] * bw)
            y_cen += int(offset[1] * bh)
        x_cen = max(bw, min(w - bw, x_cen))
        y_cen = max(bh, min(h - bh, y_cen))

        return [(x_cen - bw), (y_cen - bh), (x_cen + bw), (y_cen + bh)], scale

    def get_bbox_image(self, labels):
        labels_arr = np.array(labels)
        labels_arr = np.sum(labels_arr != 0, axis=2)
        labels_nz = np.nonzero(labels_arr)
        labels_y = labels_nz[0]
        labels_x = labels_nz[1]
        y_min, y_max = labels_y.min(), labels_y.max()
        x_min, x_max = labels_x.min(), labels_x.max()
        h, w = labels_arr.shape
        im_bbox = np.zeros((h, w), np.uint8)
        y_margin = h//16
        x_margin = w//16
        x_min = int(max(0.0, x_min - x_margin))
        x_max = int(min(w, x_max + x_margin))
        y_min = int(max(0.0, y_min - y_margin))
        y_max = int(min(h, y_max + y_margin))
        im_bbox[y_min:y_max, x_min:x_max] = 255.0
        return Image.fromarray(im_bbox)


if __name__ == "__main__":
    pose_video_json_pth = "/data/youtube-dance/clean_video_dict.json"
    msk_json_pth = "/data/youtube-dance/clean_unseen_video_dict.json"
    label_path = "/data/youtube-dance/checked_openpose"
    image_path = "/data/youtube-dance/checked_images"
    IMG_MEAN = np.array((101.84807705937696, 112.10832843463207, 111.65973036298041), dtype=np.float32)
    dataset = PoseDatasetTrainVideoMask(json_path=pose_video_json_pth,
                                        label_path=label_path,
                                        image_path=image_path,
                                        mean=IMG_MEAN,
                                        n_frame_total=10,
                                        is_jitter=True, is_mirror=True, interval=1)
    out = dataset[0]

