import numpy as np
import torch
import sys
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
import glob


global_pose_color_list = [
    [153, 0, 153], [153, 0, 102], [102, 0, 153], [51, 0, 153],  # head
    [153, 0, 51], [153, 0, 0],  # body
    [153, 51, 0], [153, 102, 0], [153, 153, 0],  # right arm
    [102, 153, 0], [51, 153, 0], [0, 153, 0],  # left arm
    [0, 153, 51], [0, 153, 102], [0, 153, 153],  # right leg
    [0, 102, 153], [0, 51, 153], [0, 0, 153],  # left leg
    ################
    [204, 0, 0], [163, 204, 0], [0, 204, 82], [0, 82, 204], [163, 0, 204],  # hand
    #################
    [255, 255, 255]  # face
]
global_pose_color_dict = {tuple(key): index+1 for index, key in enumerate(global_pose_color_list)}
global_pose_color_dict[(0, 0, 0)] = 0
global_pose_color_dict_reversed = {index+1: key for index, key in enumerate(global_pose_color_list)}


def im2vl(img, t, basic_point_only=True, remove_face_labels=True):
    if t == "face":
        img_tmp = np.zeros(img.shape, dtype=np.uint8)
        img_tmp[img == 255] = 1
        # assert np.sum(img == 255) == np.sum(img != 0)
    elif t == "pose":
        # check validity
        colors, counts = np.unique(img.reshape(-1, 3), axis=0, return_counts=1)
        # print("num_colors:", colors.shape[0])
        if basic_point_only and remove_face_labels:
            num_colors = 19
        else:
            num_colors = 25
        assert counts.size <= num_colors
        img_tmp = np.zeros(img.shape[:2], dtype=np.uint8)
        for col in colors:
            col_msk = np.all(img == col, axis=2)
            img_tmp[col_msk] = global_pose_color_dict[tuple(col)]
    else:
        raise KeyError('input is illegal!')
    return img_tmp


def vl2ch(img_tensor_batch, t, basic_point_only=False, remove_face_labels=False):
    if t == "face":
        b, h, w = img_tensor_batch.size()
        img_tmp = torch.zeros(size=(b, 2, h, w), dtype=torch.float32)
        for ci in range(2):
            img_tmp[:, ci, :, :] = (img_tensor_batch.squeeze()==ci).to(dtype=torch.float32)
    elif t == "pose":
        b, h, w = img_tensor_batch.size()
        if basic_point_only and remove_face_labels:
            num_colors = 19
        else:
            num_colors = 25
        img_tmp = torch.zeros(size=(b, num_colors, h, w), dtype=torch.float32)
        for ci in range(num_colors):
            img_tmp[:, ci, :, :] = (img_tensor_batch.squeeze() == ci).to(dtype=torch.float32)
    else:
        raise KeyError('input is illegal!')
    return img_tmp


def vl2im(img, t, basic_point_only=False, remove_face_labels=False):
    if t == "face":
        img_tmp = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        img_tmp[img == 1] = 255
    elif t == "pose":
        img_tmp = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        if basic_point_only and remove_face_labels:
            num_colors = 19
        else:
            num_colors = 25
        for i in range(1, num_colors):
            col_msk = img == i
            img_tmp[col_msk] = global_pose_color_dict_reversed[i]
    else:
        raise KeyError('input is illegal!')
    return img_tmp


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    if isinstance(image_tensor, torch.autograd.Variable):
        image_tensor = image_tensor.data
    if len(image_tensor.size()) == 5:
        image_tensor = image_tensor[0, -1]
    if len(image_tensor.size()) == 4:
        image_tensor = image_tensor[0]
    if len(image_tensor.size()) == 2:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor[:3]
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        min = np.min(image_numpy)
        max = np.max(image_numpy)
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) - min) / (max - min) * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)


def map2fig(heatmap, initial=True):
    dpi = 1000.0
    plt.ioff()
    fig = plt.figure(frameon=False)
    if initial:
        heatmap[0, 0] = 1.0
    fig.clf()
    fig.set_size_inches(heatmap.shape[1] / dpi, heatmap.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    cm = plt.cm.get_cmap('jet')
    ax.imshow(heatmap, cmap=cm, aspect='auto')
    fig.set_dpi(int(dpi))
    plt.close(fig)
    return fig2data(fig)[:, :, :3]


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def check_path_valid(A_paths, B_paths):
    assert(len(A_paths) == len(B_paths))
    for a, b in zip(A_paths, B_paths):
        assert(len(a) == len(b))


def plot_grid(x, y, ax=None, **kwargs):
    ax = ax or plt.gca()
    segs1 = np.stack((x, y), axis=2)
    segs2 = segs1.transpose(1, 0, 2)
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))
    ax.autoscale()


def grid2fig(warped_grid, grid_size=32):
    h_range = torch.linspace(-1, 1, grid_size)
    w_range = torch.linspace(-1, 1, grid_size)
    grid = torch.stack(torch.meshgrid([h_range, w_range]), -1).flip(2)
    flow_uv = grid.cpu().data.numpy()
    fig, ax = plt.subplots()
    grid_x, grid_y = warped_grid[..., 0], warped_grid[..., 1]
    plot_grid(flow_uv[..., 0], flow_uv[..., 1], ax=ax, color="lightgrey")
    plot_grid(grid_x, grid_y, ax=ax, color="C0")
    plt.axis("off")
    plt.tight_layout(pad=0)
    fig.set_size_inches(256/100, 256/100)
    fig.set_dpi(100)
    out = fig2data(fig)[:, :, :3]
    plt.close()
    plt.cla()
    plt.clf()
    return out


if __name__ == "__main__":
    pass
