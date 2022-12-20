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
            start_idx = random.choice(list(range(0, len(L_paths)-self.n_frame_total+1)))
        else:
            start_idx = random.choice(list(range(0, self.n_frame_total)))
        # reference frame, i.e. first frame
        anchor_ky = self.read_data(L_paths[start_idx % len(L_paths)], data_type="np")
        anchor_crop_coords, anchor_scale = self.get_crop_coords(keypoints=anchor_ky)
        anchor_bw = max(1, (anchor_crop_coords[1] - anchor_crop_coords[0]) // 256)

        source_img_list = []
        for i in range(self.n_frame_total):
            source_img = self.crop(self.read_data(I_paths[(start_idx+i) % len(L_paths)]), anchor_crop_coords)
            source_img_list.append(source_img)

        # update keypoints
        source_lbl_list = []
        source_ky_list = []
        source_bbox_list = []
        for i in range(self.n_frame_total):
            source_ky = self.read_keypoints(L_paths[(start_idx+i) % len(L_paths)], anchor_crop_coords)
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
            src_name_list.append(names[(start_idx+i) % len(L_paths)])

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
        x_margin = w//16
        y_margin = h//16
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
