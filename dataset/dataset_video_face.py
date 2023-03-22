# build a framework similar to vid2vid
# consistently use previous frame to synthesis current frame
import warnings

warnings.filterwarnings("ignore")
from PIL import Image
import numpy as np
import random
import torch.utils.data as data
from utils.keypoint2img import interp_points, draw_edge
import os
import cv2
from skimage.transform import resize
from skimage import img_as_bool
import torchvision.transforms.functional as F
from utils.misc import check_path_valid


# training which will return a bounding box for reducing the computation complexity of attention matrix
# MS: multi-source
# mask norm (MN) for all images
class FaceDatasetTrainVideoMask(data.Dataset):
    def __init__(self, label_path, image_path, mean,
                 n_frame_total,
                 is_jitter, is_mirror):
        super(FaceDatasetTrainVideoMask, self).__init__()
        self.n_frame_total = n_frame_total  # the total number of frames
        self.mean = mean
        self.is_jitter = is_jitter
        self.is_mirror = is_mirror
        self.img_size = (256, 256)
        # mapping from keypoints to face part
        self.part_list = [[list(range(0, 17))],  # face
                          [range(17, 22)],  # right eyebrow
                          [range(22, 27)],  # left eyebrow
                          [[28, 31], range(31, 36), [35, 28]],  # nose
                          [[36, 37, 38, 39], [39, 40, 41, 36]],  # right eye
                          [[42, 43, 44, 45], [45, 46, 47, 42]],  # left eye
                          [range(48, 55), [54, 55, 56, 57, 58, 59, 48], range(60, 65), [64, 65, 66, 67, 60]],
                          # mouth and tongue
                          ]
        self.lbl_pths = []
        self.img_pths = []
        self.names = []
        lbl_dirs = os.listdir(label_path)
        lbl_dirs.sort()
        img_dirs = os.listdir(image_path)
        img_dirs.sort()
        for lbl_dir_name in lbl_dirs:
            lbl_dir_pth = os.path.join(label_path, lbl_dir_name)
            lbl_dir = os.listdir(lbl_dir_pth)
            lbl_dir.sort()
            lbl_pth = [os.path.join(lbl_dir_pth, lbl_name) for lbl_name in lbl_dir]
            self.lbl_pths.append(lbl_pth)
            self.names.append(lbl_dir)
        for img_dir_name in img_dirs:
            img_dir_pth = os.path.join(image_path, img_dir_name)
            img_dir = os.listdir(img_dir_pth)
            img_dir.sort()
            img_pth = [os.path.join(img_dir_pth, img_name) for img_name in img_dir]
            self.img_pths.append(img_pth)
        check_path_valid(self.lbl_pths, self.img_pths)

    def __getitem__(self, index):
        # apply random crop/scale/flip
        # if flip, all source frames should flip
        seq_idx = index % len(self.lbl_pths)
        L_paths = self.lbl_pths[seq_idx]
        I_paths = self.img_pths[seq_idx]
        names = self.names[seq_idx]
        # random choose n_frame_total frames from the video
        if len(L_paths) > self.n_frame_total:
            start_idx = random.choice(list(range(0, len(L_paths) - self.n_frame_total + 1)))
        else:
            start_idx = random.choice(list(range(0, self.n_frame_total)))
        # reference frame, i.e. first frame
        anchor_ky = self.read_data(L_paths[start_idx % len(L_paths)], data_type="np")
        anchor_crop_coords, anchor_scale = self.get_crop_coords(keypoints=anchor_ky)
        anchor_bw = max(1, (anchor_crop_coords[1] - anchor_crop_coords[0]) // 256)

        source_img_list = []
        for i in range(self.n_frame_total):
            source_img = self.crop(self.read_data(I_paths[(start_idx + i) % len(L_paths)]), anchor_crop_coords)
            source_img_list.append(source_img)

        # update keypoints
        source_lbl_list = []
        source_ky_list = []
        source_bbox_list = []
        for i in range(self.n_frame_total):
            source_ky = self.read_keypoints(L_paths[(start_idx + i) % len(L_paths)], anchor_crop_coords)
            source_ky_list.append(source_ky)
            source_lbl = self.get_face_image(source_ky, source_img_list[i].size, bw=anchor_bw)
            source_lbl_list.append(source_lbl)
            source_bbox = self.get_bbox_image(source_ky, source_img_list[i].size)
            source_bbox_list.append(source_bbox)

        # resize to (256, 256)
        src_img_list = []
        src_lbl_list = []
        src_bbox_list = []
        for i in range(self.n_frame_total):
            src_img = source_img_list[i].resize(self.img_size)
            src_lbl = np.asarray(img_as_bool(resize(source_lbl_list[i], self.img_size)), dtype=np.uint8)  # 0 and 1
            src_lbl = Image.fromarray(src_lbl)
            src_bbox = np.asarray(img_as_bool(resize(source_bbox_list[i], self.img_size)), dtype=np.uint8)  # 0 and 1
            src_bbox = Image.fromarray(src_bbox)
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
        src_bbox_arr_list = [np.asarray(src_bbox, dtype=np.uint8) for src_bbox in src_bbox_list]

        src_img_arr_list = [cv2.cvtColor(np.asarray(src_img), cv2.COLOR_RGB2BGR) for src_img in src_img_list]
        src_img_arr_list = [np.asarray(src_img_arr, np.float32) for src_img_arr in src_img_arr_list]

        src_img_arr_list = src_img_arr_list - self.mean
        src_img_arr_list = [src_img_arr.transpose((2, 0, 1)) for src_img_arr in src_img_arr_list]

        src_img_arr_list = [src_img_arr.copy() for src_img_arr in src_img_arr_list]
        src_lbl_arr_list = [src_lbl_arr.copy() for src_lbl_arr in src_lbl_arr_list]
        src_bbox_arr_list = [src_bbox_arr.copy() for src_bbox_arr in src_bbox_arr_list]

        src_name_list = []
        for i in range(self.n_frame_total):
            src_name_list.append(names[(start_idx + i) % len(L_paths)])

        return src_img_arr_list, src_lbl_arr_list, src_bbox_arr_list, src_name_list

    def read_data(self, path, data_type='img'):
        is_img = data_type == 'img'
        if is_img:
            img = Image.open(path)
        elif data_type == 'np':
            img = np.loadtxt(path, delimiter=',')
        else:
            img = path
        return img

    def get_face_image(self, keypoints, size, bw):
        w, h = size
        edge_len = 3  # interpolate 3 keypoints to form a curve when drawing edges
        # edge map for face region from keypoints
        im_edges = np.zeros((h, w), np.uint8)  # edge map for all edges
        for edge_list in self.part_list:
            for edge in edge_list:
                for i in range(0, max(1, len(edge) - 1),
                               edge_len - 1):  # divide a long edge into multiple small edges when drawing
                    sub_edge = edge[i:i + edge_len]
                    x = keypoints[sub_edge, 0]
                    y = keypoints[sub_edge, 1]

                    curve_x, curve_y = interp_points(x, y)  # interp keypoints to get the curve shape
                    draw_edge(im_edges, curve_x, curve_y, bw=bw)
        return im_edges

    def get_bbox_image(self, keypoints, size):
        w, h = size
        im_bbox = np.zeros((h, w), np.uint8)
        x_min = keypoints[:, 0].min()
        x_max = keypoints[:, 0].max()
        y_min = keypoints[:, 1].min()
        y_max = keypoints[:, 1].max()
        x_margin = w // 16
        y_margin = h // 16
        x_min = int(max(0.0, x_min - x_margin))
        x_max = int(min(w, x_max + x_margin))
        y_min = int(max(0.0, y_min - y_margin))
        y_max = int(min(h, y_max + y_margin))
        im_bbox[y_min:y_max, x_min:x_max] = 255.0
        return im_bbox

    def read_keypoints(self, L_path, crop_coords):
        keypoints = self.read_data(L_path, data_type='np')

        if crop_coords is None:
            crop_coords, _ = self.get_crop_coords(keypoints)
        keypoints[:, 0] -= crop_coords[2]
        keypoints[:, 1] -= crop_coords[0]

        return keypoints

    def get_crop_coords(self, keypoints, scale=None):
        min_y, max_y = int(keypoints[:, 1].min()), int(keypoints[:, 1].max())
        min_x, max_x = int(keypoints[:, 0].min()), int(keypoints[:, 0].max())
        x_cen, y_cen = (min_x + max_x) // 2, (min_y + max_y) // 2
        w = h = (max_x - min_x)
        offset_max = 0.2
        offset = [random.uniform(-offset_max, offset_max),
                  random.uniform(-offset_max, offset_max)]

        scale_max = 0.2
        if scale is None:
            scale = [random.uniform(1 - scale_max, 1 + scale_max),
                     random.uniform(1 - scale_max, 1 + scale_max)]
        w *= scale[0]
        h *= scale[1]
        x_cen += int(offset[0] * w)
        y_cen += int(offset[1] * h)

        min_x = x_cen - w
        min_y = y_cen - h * 1.25
        max_x = min_x + w * 2
        max_y = min_y + h * 2

        return [int(min_y), int(max_y), int(min_x), int(max_x)], scale

    def crop(self, img, coords):
        min_y, max_y, min_x, max_x = coords
        if isinstance(img, np.ndarray):
            return img[min_y:max_y, min_x:max_x]
        else:
            return img.crop((min_x, min_y, max_x, max_y))

    def __len__(self):
        return len(self.lbl_pths)


def read_dir(dir_path):
    file_name_list = os.listdir(dir_path)
    file_name_list.sort()
    file_path_list = [os.path.join(dir_path, x) for x in file_name_list]
    return file_path_list


class FaceDatasetTest(data.Dataset):
    def __init__(self,
                 sub_images_path,
                 sub_labels_path,
                 dri_images_path,
                 dri_labels_path,
                 mean, fix_crop_pos=True,
                 max_frame_num=None):
        super(FaceDatasetTest, self).__init__()
        self.max_frame_num = max_frame_num

        self.sub_images_path = sub_images_path
        self.sub_labels_path = sub_labels_path
        self.dri_images_path = dri_images_path
        self.dri_labels_path = dri_labels_path

        self.mean = mean
        self.fix_crop_pos = fix_crop_pos
        self.ref_dist_x, self.ref_dist_y = [None] * 83, [None] * 83
        self.dist_scale_x, self.dist_scale_y = [None] * 83, [None] * 83
        self.img_size = (256, 256)
        # mapping from keypoints to face part
        self.part_list = [[list(range(0, 17))],  # face
                          [range(17, 22)],  # right eyebrow
                          [range(22, 27)],  # left eyebrow
                          [[28, 31], range(31, 36), [35, 28]],  # nose
                          [[36, 37, 38, 39], [39, 40, 41, 36]],  # right eye
                          [[42, 43, 44, 45], [45, 46, 47, 42]],  # left eye
                          [range(48, 55), [54, 55, 56, 57, 58, 59, 48], range(60, 65), [64, 65, 66, 67, 60]],
                          # mouth and tongue
                          ]

        self.files = []
        self.files.append({
            "src_vid_dir": sub_images_path,
            "src_lbl_dir": sub_labels_path,
            "tar_vid_dir": dri_images_path,
            "tar_lbl_dir": dri_labels_path
        })

    def __getitem__(self, index):
        # generate an image with the same mask as target label but the same person as source image
        datafiles = self.files[index]
        # read source video
        src_ky_list = os.listdir(datafiles["src_lbl_dir"])
        src_ky_list.sort()
        if self.max_frame_num is not None:
            max_frame_num = min(len(src_ky_list), self.max_frame_num)
            src_ky_list = src_ky_list[:max_frame_num]

        # read first frame
        src_ky = self.read_data(os.path.join(datafiles["src_lbl_dir"], src_ky_list[0]),
                                data_type="np")
        src_crop_coords = self.get_crop_coords(keypoints=src_ky)
        src_bw = max(1, (src_crop_coords[1] - src_crop_coords[0]) // 256)
        if not self.fix_crop_pos:
            src_crop_coords_list = [self.get_crop_coords(self.read_data(os.path.join(datafiles["src_lbl_dir"], ky_name),
                                                                        data_type="np")) for ky_name in src_ky_list]
        src_crop_ky_list = [self.read_keypoints(os.path.join(datafiles["src_lbl_dir"], ky_name),
                                                src_crop_coords if self.fix_crop_pos else None) for ky_name in
                            src_ky_list]

        src_img_list = []
        src_lbl_list = []
        src_bbox_list = []
        src_name_list = []
        for idx, src_crop_ky in enumerate(src_crop_ky_list):
            src_img_name = src_ky_list[idx].replace(".txt", ".png")
            src_img_file = os.path.join(datafiles["src_vid_dir"], src_img_name)
            if self.fix_crop_pos:
                src_img = self.crop(self.read_data(src_img_file), src_crop_coords)
            else:
                src_img = self.crop(self.read_data(src_img_file), src_crop_coords_list[idx])
            # update keypoints & mask norm
            src_lbl = self.get_face_image(src_crop_ky, src_img.size, bw=src_bw)
            src_bbox = self.get_bbox_image(src_crop_ky, src_img.size)
            src_img = src_img.resize(self.img_size)
            src_lbl = np.asarray(img_as_bool(resize(src_lbl, self.img_size)), dtype=np.uint8)
            src_bbox = np.asarray(img_as_bool(resize(src_bbox, self.img_size)), dtype=np.uint8)
            src_img_arr = cv2.cvtColor(np.asarray(src_img), cv2.COLOR_RGB2BGR)
            src_img_arr = np.asarray(src_img_arr, np.float32)
            src_img_arr -= self.mean
            src_img_arr = src_img_arr.transpose((2, 0, 1))
            src_bbox_list.append(src_bbox)
            src_lbl_list.append(src_lbl)
            src_img_list.append(src_img_arr)
            src_name_list.append(src_img_name)
        self.normalize_faces(src_crop_ky_list, is_ref=True)

        # read target video
        tar_ky_list = os.listdir(datafiles["tar_lbl_dir"])
        tar_ky_list.sort()
        if self.max_frame_num is not None:
            max_frame_num = min(len(tar_ky_list), self.max_frame_num)
            tar_ky_list = tar_ky_list[:max_frame_num]
        # read first frame
        tar_ky = self.read_data(os.path.join(datafiles["tar_lbl_dir"], tar_ky_list[0]),
                                data_type="np")
        tar_crop_coords = self.get_crop_coords(keypoints=tar_ky)
        tar_bw = max(1, (tar_crop_coords[1] - tar_crop_coords[0]) // 256)
        if not self.fix_crop_pos:
            tar_crop_coords_list = [self.get_crop_coords(self.read_data(os.path.join(datafiles["tar_lbl_dir"], ky_name),
                                                                        data_type="np")) for ky_name in tar_ky_list]
        tar_crop_ky_list = [self.read_keypoints(os.path.join(datafiles["tar_lbl_dir"], ky_name),
                                                tar_crop_coords if self.fix_crop_pos else None) for ky_name in
                            tar_ky_list]

        tar_crop_ky_list = self.normalize_faces(tar_crop_ky_list, is_ref=False)

        new_tar_crop_ky_list = []
        # moving average
        tar_crop_ky_list_npy = np.stack(tar_crop_ky_list, axis=0)
        num_ky = tar_crop_ky_list_npy.shape[1]
        for iky in range(num_ky):
            cur_tar_crop_ky_list_npy = tar_crop_ky_list_npy[:, iky, :]
            tar_crop_ky_cumsum = np.cumsum(cur_tar_crop_ky_list_npy, axis=0)
            win_len = 5
            assert win_len == 5
            num_frame = cur_tar_crop_ky_list_npy.shape[0]
            new_tar_crop_ky_npy = np.zeros_like(cur_tar_crop_ky_list_npy)
            new_tar_crop_ky_npy[0] = tar_crop_ky_cumsum[0]
            new_tar_crop_ky_npy[1] = tar_crop_ky_cumsum[2] / 3
            new_tar_crop_ky_npy[2] = tar_crop_ky_cumsum[4] / 5
            for ii in range(3, num_frame - 2):
                new_tar_crop_ky_npy[ii] = (tar_crop_ky_cumsum[ii + 2] - tar_crop_ky_cumsum[ii - 3]) / win_len
            new_tar_crop_ky_npy[num_frame - 2] = (tar_crop_ky_cumsum[-1] - tar_crop_ky_cumsum[-4]) / 3
            new_tar_crop_ky_npy[num_frame - 1] = cur_tar_crop_ky_list_npy[-1]
            new_tar_crop_ky_list.append(new_tar_crop_ky_npy)

        new_tar_crop_ky_list_npy = np.stack(new_tar_crop_ky_list, axis=1)
        tar_crop_ky_list = new_tar_crop_ky_list_npy.tolist()
        tar_crop_ky_list = [np.array(x) for x in tar_crop_ky_list]

        tar_img_list = []
        tar_lbl_list = []
        tar_bbox_list = []
        tar_name_list = []

        # normalize target keypoints
        for idx, tar_crop_ky in enumerate(tar_crop_ky_list):
            tar_img_name = tar_ky_list[idx].replace(".txt", ".png")
            tar_img_file = os.path.join(datafiles["tar_vid_dir"], tar_img_name)
            if self.fix_crop_pos:
                tar_img = self.crop(self.read_data(tar_img_file), tar_crop_coords)
            else:
                tar_img = self.crop(self.read_data(tar_img_file), tar_crop_coords_list[idx])
            tar_lbl = self.get_face_image(tar_crop_ky, tar_img.size, bw=tar_bw)
            tar_bbox = self.get_bbox_image(tar_crop_ky, tar_img.size)
            tar_img = tar_img.resize(self.img_size)
            tar_lbl = np.asarray(img_as_bool(resize(tar_lbl, self.img_size)), dtype=np.uint8)
            tar_bbox = np.asarray(img_as_bool(resize(tar_bbox, self.img_size)), dtype=np.uint8)
            tar_img_arr = cv2.cvtColor(np.asarray(tar_img), cv2.COLOR_RGB2BGR)
            tar_img_arr = np.asarray(tar_img_arr, np.float32)
            tar_img_arr -= self.mean
            tar_img_arr = tar_img_arr.transpose((2, 0, 1))
            tar_bbox_list.append(tar_bbox)
            tar_lbl_list.append(tar_lbl)
            tar_img_list.append(tar_img_arr)
            tar_name_list.append(tar_img_name)

        return np.stack(src_img_list, axis=0), np.stack(src_lbl_list, axis=0), np.stack(src_bbox_list), src_name_list, \
               np.stack(tar_img_list, axis=0), np.stack(tar_lbl_list, axis=0), np.stack(tar_bbox_list), tar_name_list

    def normalize_faces(self, all_keypoints, is_ref=False):
        central_keypoints = [8]
        face_centers = [np.mean(keypoints[central_keypoints, :], axis=0) for keypoints in all_keypoints]
        compute_mean = not is_ref
        if compute_mean:
            img_scale = self.img_scale / (all_keypoints[0][:, 0].max() - all_keypoints[0][:, 0].min())

        part_list = [[0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10], [7, 9, 8],  # face 17
                     [17, 26], [18, 25], [19, 24], [20, 23], [21, 22],  # eyebrows 10
                     [27], [28], [29], [30], [31, 35], [32, 34], [33],  # nose 9
                     [36, 45], [37, 44], [38, 43], [39, 42], [40, 47], [41, 46],  # eyes 12
                     [48, 54], [49, 53], [50, 52], [51], [55, 59], [56, 58], [57],  # mouth 12
                     [60, 64], [61, 63], [62], [65, 67], [66],  # tongue 8
                     ]

        for i, pts_idx in enumerate(part_list):
            if compute_mean or is_ref:
                mean_dists_x, mean_dists_y = [], []
                for k, keypoints in enumerate(all_keypoints):
                    pts = keypoints[pts_idx]
                    pts_cen = np.mean(pts, axis=0)
                    face_cen = face_centers[k]
                    for p, pt in enumerate(pts):
                        mean_dists_x.append(np.linalg.norm(pt - pts_cen))
                        mean_dists_y.append(np.linalg.norm(pts_cen - face_cen))
                mean_dist_x = sum(mean_dists_x) / len(mean_dists_x) + 1e-3
                mean_dist_y = sum(mean_dists_y) / len(mean_dists_y) + 1e-3
            if is_ref:
                self.ref_dist_x[i] = mean_dist_x
                self.ref_dist_y[i] = mean_dist_y
                self.img_scale = all_keypoints[0][:, 0].max() - all_keypoints[0][:, 0].min()
            else:
                if compute_mean:
                    self.dist_scale_x[i] = self.ref_dist_x[i] / mean_dist_x / img_scale
                    self.dist_scale_y[i] = self.ref_dist_y[i] / mean_dist_y / img_scale

                for k, keypoints in enumerate(all_keypoints):
                    pts = keypoints[pts_idx]
                    pts_cen = np.mean(pts, axis=0)
                    face_cen = face_centers[k]
                    pts = (pts - pts_cen) * self.dist_scale_x[i] + (pts_cen - face_cen) * self.dist_scale_y[
                        i] + face_cen
                    all_keypoints[k][pts_idx] = pts
        return all_keypoints

    def read_data(self, path, data_type='img'):
        is_img = data_type == 'img'
        if is_img:
            img = Image.open(path)
        elif data_type == 'np':
            img = np.loadtxt(path, delimiter=',')
        else:
            img = path
        return img

    def get_face_image(self, keypoints, size, bw):
        w, h = size
        edge_len = 3  # interpolate 3 keypoints to form a curve when drawing edges
        # edge map for face region from keypoints
        im_edges = np.zeros((h, w), np.uint8)  # edge map for all edges
        for edge_list in self.part_list:
            for edge in edge_list:
                for i in range(0, max(1, len(edge) - 1),
                               edge_len - 1):  # divide a long edge into multiple small edges when drawing
                    sub_edge = edge[i:i + edge_len]
                    x = keypoints[sub_edge, 0]
                    y = keypoints[sub_edge, 1]

                    curve_x, curve_y = interp_points(x, y)  # interp keypoints to get the curve shape
                    draw_edge(im_edges, curve_x, curve_y, bw=bw)
        return im_edges

    def get_bbox_image(self, keypoints, size):
        w, h = size
        im_bbox = np.zeros((h, w), np.uint8)
        x_min = keypoints[:, 0].min()
        x_max = keypoints[:, 0].max()
        y_min = keypoints[:, 1].min()
        y_max = keypoints[:, 1].max()
        x_margin = w // 16
        y_margin = h // 16
        x_min = int(max(0.0, x_min - x_margin))
        x_max = int(min(w, x_max + x_margin))
        y_min = int(max(0.0, y_min - y_margin))
        y_max = int(min(h, y_max + y_margin))
        im_bbox[y_min:y_max, x_min:x_max] = 255.0
        return im_bbox

    def read_keypoints(self, L_path, crop_coords):
        keypoints = self.read_data(L_path, data_type='np')

        if crop_coords is None:
            crop_coords = self.get_crop_coords(keypoints)
        keypoints[:, 0] -= crop_coords[2]
        keypoints[:, 1] -= crop_coords[0]

        return keypoints

    def get_crop_coords(self, keypoints):
        min_y, max_y = int(keypoints[:, 1].min()), int(keypoints[:, 1].max())
        min_x, max_x = int(keypoints[:, 0].min()), int(keypoints[:, 0].max())
        x_cen, y_cen = (min_x + max_x) // 2, (min_y + max_y) // 2
        w = h = (max_x - min_x)

        min_x = x_cen - w
        min_y = y_cen - h * 1.25
        max_x = min_x + w * 2
        max_y = min_y + h * 2

        return int(min_y), int(max_y), int(min_x), int(max_x)

    def crop(self, img, coords):
        min_y, max_y, min_x, max_x = coords
        if isinstance(img, np.ndarray):
            return img[min_y:max_y, min_x:max_x]
        else:
            return img.crop((min_x, min_y, max_x, max_y))

    def __len__(self):
        return len(self.files)


if __name__ == "__main__":
    IMG_MEAN = np.array((101.84807705937696, 112.10832843463207, 111.65973036298041), dtype=np.float32)
    label_path = "/data/FaceForensics/clean_keypoints"
    image_path = "/data/FaceForensics/clean_sampled_frames"
    dataset = FaceDatasetTrainVideoMask(n_frame_total=10,
                                        label_path=label_path,
                                        image_path=image_path,
                                        is_jitter=True,
                                        is_mirror=True,
                                        mean=IMG_MEAN)
    trainloadermix = data.DataLoader(dataset, batch_size=2,
                                     shuffle=True, num_workers=1)
    for i, data in enumerate(trainloadermix):
        pass
