import numpy as np
import json
import random
from scipy.optimize import curve_fit
import warnings
import math
from copy import deepcopy


# Read json file and returns the drawn image.
def read_keypoints_posenorm(opt, json_input, size, basic_point_only=False, remove_face_labels=False,
                            ppl_idx=None, ref_pts_length=None):
    if type(json_input) == str:
        with open(json_input, encoding='utf-8') as f:
            keypoint_dicts = json.loads(f.read())["people"]
    else:
        keypoint_dicts = json.loads(json_input)["people"]

    edge_lists = define_edge_lists(basic_point_only)
    w, h = size
    pose_img = np.zeros((h, w, 3), np.uint8)
    pose_keypoints = np.zeros((25, 3))
    y_len_max = 0

    if ppl_idx is not None:
        keypoint_dicts = [keypoint_dicts[ppl_idx]]
    for keypoint_dict in keypoint_dicts:
        pose_pts = np.array(keypoint_dict["pose_keypoints_2d"]).reshape(25, 3)
        face_pts = np.array(keypoint_dict["face_keypoints_2d"]).reshape(70, 3)
        hand_pts_l = np.array(keypoint_dict["hand_left_keypoints_2d"]).reshape(21, 3)
        hand_pts_r = np.array(keypoint_dict["hand_right_keypoints_2d"]).reshape(21, 3)
        pts = [extract_valid_keypoints(pts, edge_lists) for pts in [pose_pts, face_pts, hand_pts_l, hand_pts_r]]

        pose_pts = pts[0]
        x, y = pose_pts[:,0], pose_pts[:,1]
        y_len = y.max() - y.min()
        if y_len > y_len_max:
            y_len_max = y_len
            pose_img = connect_keypoints(opt, pts, edge_lists, size, basic_point_only, remove_face_labels)
            pose_keypoints = pose_pts
    return pose_img, pose_keypoints, pts


def read_keypoints_posenorm_smooth(opt, keypoint_dict, index, size, basic_point_only=False, remove_face_labels=False,
                                   ppl_idx=None, ref_pts_length=None):
    edge_lists = define_edge_lists(basic_point_only)
    w, h = size
    pose_img = np.zeros((h, w, 3), np.uint8)
    pose_keypoints = np.zeros((25, 3))
    y_len_max = 0
    pose_pts = keypoint_dict["pose_keypoints_2d"][index]
    face_pts = keypoint_dict["face_keypoints_2d"][index]
    hand_pts_l = keypoint_dict["hand_left_keypoints_2d"][index]
    hand_pts_r = keypoint_dict["hand_right_keypoints_2d"][index]
    pts = [pose_pts, face_pts, hand_pts_l, hand_pts_r]

    pose_pts = pts[0]
    x, y = pose_pts[:,0], pose_pts[:,1]
    y_len = y.max() - y.min()
    if y_len > y_len_max:
        y_len_max = y_len
        pose_img = connect_keypoints(opt, pts, edge_lists, size, basic_point_only, remove_face_labels)
        pose_keypoints = pose_pts
    return pose_img, pose_keypoints, pts


def read_pts_posenorm(opt, pts, crop_coords, size, basic_point_only=False, remove_face_labels=False,
                      ref_pts_length=False):
    # reset the coordinate of pts
    xs, ys, xe, ye = crop_coords
    for c_idx, coord_list in enumerate(pts):
        for i_idx, coord in enumerate(coord_list):
            if 0 not in coord:
                pts[c_idx][i_idx] = coord - np.array([xs, ys])

    edge_lists = define_edge_lists(basic_point_only)
    w, h = size
    pose_img = np.zeros((h, w, 3), np.uint8)
    pose_keypoints = np.zeros((25, 3))
    y_len_max = 0

    pose_edge_list = edge_lists[0]
    hand_edge_connections = edge_lists[2]
    hand_edge_list = []
    for hand_edge_connection in hand_edge_connections:
        num_part = len(hand_edge_connection) - 1
        for i in range(num_part):
            hand_edge_list.append([hand_edge_connection[i], hand_edge_connection[i + 1]])
    hand_dict = {2: 7, 3: 4}

    if ref_pts_length:
        new_pts = deepcopy(pts)
        cur_pose_pts = pts[0]
        new_cur_pose_pts = deepcopy(cur_pose_pts)
        cur_pose_pts_length = []
        for edge_part in pose_edge_list:
            if (0 in cur_pose_pts[edge_part[0]]) or (0 in cur_pose_pts[edge_part[1]]):
                edge_part_length = 0
            else:
                edge_part_length = np.linalg.norm(cur_pose_pts[edge_part[0]] - cur_pose_pts[edge_part[1]])
            cur_pose_pts_length.append(edge_part_length)
        cur_pose_pts_length = np.array(cur_pose_pts_length)
        cur_torso_edge_part_length = cur_pose_pts_length[5]
        if ref_pts_length == "fm":
            new_cur_torso_edge_part_length = cur_torso_edge_part_length * 0.85
        if ref_pts_length == "mf":
            new_cur_torso_edge_part_length = cur_torso_edge_part_length * 1.2
        # change shoulder
        # compute the new coordinate of 2, 5
        cur_anchor_pose_coord = cur_pose_pts[1]
        for i in [2, 5]:
            cur_pose_coord = cur_pose_pts[i]
            if 0 in cur_pose_coord:
                continue
            cur_pose_vector = cur_pose_coord - cur_anchor_pose_coord
            if ref_pts_length == "fm":
                new_cur_pose_coord = new_cur_pose_pts[1] + cur_pose_vector * 0.9
            if ref_pts_length == "mf":
                new_cur_pose_coord = new_cur_pose_pts[1] + cur_pose_vector * 1.2
            new_cur_pose_pts[i] = new_cur_pose_coord
        inner_pose_points_list = [[2, 5], [3, 6]]
        outer_pose_points_list = [[3, 6], [4, 7]]
        for j in range(len(inner_pose_points_list)):
            inner_pose_points = inner_pose_points_list[j]
            outer_pose_points = outer_pose_points_list[j]
            for anchor_point in inner_pose_points:
                anchor_pose_coord = new_cur_pose_pts[anchor_point]
                cur_anchor_pose_coord = cur_pose_pts[anchor_point]
                for point in outer_pose_points:
                    cur_pose_coord = cur_pose_pts[point]
                    # compute the unit vector
                    cur_pose_vector = cur_pose_coord - cur_anchor_pose_coord
                    if [anchor_point, point] in pose_edge_list:
                        cur_edge_part = [anchor_point, point]
                    elif [point, anchor_point] in pose_edge_list:
                        cur_edge_part = [point, anchor_point]
                    else:
                        continue
                    cur_edge_part_idx = pose_edge_list.index(cur_edge_part)
                    if cur_pose_pts_length[cur_edge_part_idx]:
                        new_cur_pose_coord = anchor_pose_coord + cur_pose_vector
                        new_cur_pose_pts[point] = new_cur_pose_coord

        # change the torso and leg
        # compute the new coordinate of 8
        cur_anchor_pose_coord = cur_pose_pts[1]
        cur_pose_coord = cur_pose_pts[8]
        cur_pose_vector = cur_pose_coord - cur_anchor_pose_coord
        new_cur_pose_coord = cur_anchor_pose_coord + new_cur_torso_edge_part_length * (cur_pose_vector/cur_pose_pts_length[5])
        new_cur_pose_pts[8] = new_cur_pose_coord
        # compute the new coordinate of 9, 12
        cur_anchor_pose_coord = cur_pose_pts[8]
        for i in [9, 12]:
            cur_pose_coord = cur_pose_pts[i]
            cur_pose_vector = cur_pose_coord - cur_anchor_pose_coord
            new_cur_pose_coord = new_cur_pose_pts[8] + cur_pose_vector
            new_cur_pose_pts[i] = new_cur_pose_coord
        # compute the new coordinate of 10, 13
        inner_pose_points = [9, 12]
        outer_pose_points = [10, 13]
        for anchor_point in inner_pose_points:
            anchor_pose_coord = new_cur_pose_pts[anchor_point]
            cur_anchor_pose_coord = cur_pose_pts[anchor_point]
            for point in outer_pose_points:
                cur_pose_coord = cur_pose_pts[point]
                # compute the unit vector
                cur_pose_vector = cur_pose_coord - cur_anchor_pose_coord
                if [anchor_point, point] in pose_edge_list:
                    cur_edge_part = [anchor_point, point]
                elif [point, anchor_point] in pose_edge_list:
                    cur_edge_part = [point, anchor_point]
                else:
                    continue
                cur_edge_part_idx = pose_edge_list.index(cur_edge_part)
                if cur_pose_pts_length[cur_edge_part_idx]:
                    new_cur_edge_length = (h - anchor_pose_coord[1]) * \
                                          (cur_pose_pts_length[cur_edge_part_idx]/(h - cur_anchor_pose_coord[1]))
                    new_cur_pose_coord = anchor_pose_coord + new_cur_edge_length * \
                                         (cur_pose_vector / cur_pose_pts_length[cur_edge_part_idx])
                    new_cur_pose_pts[point] = new_cur_pose_coord
        new_pts[0] = deepcopy(new_cur_pose_pts)

        # hand normalization
        for hand_idx in [2, 3]:
            cur_hand_pts = pts[hand_idx]
            cur_hand_pts_length = []
            for edge_part in hand_edge_list:
                if (0 in cur_hand_pts[edge_part[0]]) or (0 in cur_hand_pts[edge_part[1]]):
                    edge_part_length = 0
                else:
                    edge_part_length = np.linalg.norm(cur_hand_pts[edge_part[0]] - cur_hand_pts[edge_part[1]])
                cur_hand_pts_length.append(edge_part_length)
            cur_hand_pts_length = np.array(cur_hand_pts_length)
            # rearrange the coordinate
            new_cur_hand_pts = deepcopy(cur_hand_pts)
            # left 2, right 3, left 7, right 4
            new_cur_hand_pts[0] = new_pts[0][hand_dict[hand_idx]]
            inner_hand_points_list = [[0], [1, 5, 9, 13, 17], [2, 6, 10, 14, 18], [3, 7, 11, 15, 19]]
            outer_hand_points_list = [[1, 5, 9, 13, 17], [2, 6, 10, 14, 18], [3, 7, 11, 15, 19], [4, 8, 12, 16, 20]]
            for j in range(len(inner_hand_points_list)):
                inner_hand_points = inner_hand_points_list[j]
                outer_hand_points = outer_hand_points_list[j]
                for anchor_point in inner_hand_points:
                    anchor_hand_coord = new_cur_hand_pts[anchor_point]
                    cur_anchor_hand_coord = cur_hand_pts[anchor_point]
                    for point in outer_hand_points:
                        cur_hand_coord = cur_hand_pts[point]
                        # compute the unit vector
                        cur_hand_vector = cur_hand_coord - cur_anchor_hand_coord
                        if [anchor_point, point] in hand_edge_list:
                            cur_edge_part = [anchor_point, point]
                        elif [point, anchor_point] in hand_edge_list:
                            cur_edge_part = [point, anchor_point]
                        else:
                            continue
                        cur_edge_part_idx = hand_edge_list.index(cur_edge_part)
                        if cur_hand_pts_length[cur_edge_part_idx]:
                            new_cur_hand_coord = anchor_hand_coord + cur_hand_vector
                            new_cur_hand_pts[point] = new_cur_hand_coord
            new_pts[hand_idx] = deepcopy(new_cur_hand_pts)

        new_pose_pts = new_pts[0]
        x, y = new_pose_pts[:, 0], new_pose_pts[:, 1]
        y_len = y.max() - y.min()
        if y_len > y_len_max:
            y_len_max = y_len
            new_pose_img = connect_keypoints(opt, new_pts, edge_lists, size, basic_point_only, remove_face_labels)
            new_pose_keypoints = new_pose_pts
        return new_pose_img, new_pose_keypoints, new_pts
    else:
        pose_pts = pts[0]
        x, y = pose_pts[:, 0], pose_pts[:, 1]
        y_len = y.max() - y.min()
        if y_len > y_len_max:
            y_len_max = y_len
            pose_img = connect_keypoints(opt, pts, edge_lists, size, basic_point_only, remove_face_labels)
            pose_keypoints = pose_pts
        return pose_img, pose_keypoints, pts


# Use only the valid keypoints in the list.
def extract_valid_keypoints(pts, edge_lists):
    pose_edge_list, _, hand_edge_list, _, face_list = edge_lists
    p = pts.shape[0]
    thre = 0.1 if p == 70 else 0.01
    output = np.zeros((p, 2))

    if p == 70:  # face
        for edge_list in face_list:
            for edge in edge_list:
                if (pts[edge, 2] > thre).all():
                    output[edge, :] = pts[edge, :2]
    elif p == 21:  # hand
        for edge in hand_edge_list:
            if (pts[edge, 2] > thre).all():
                output[edge, :] = pts[edge, :2]
    else:  # pose
        valid = (pts[:, 2] > thre)
        output[valid, :] = pts[valid, :2]

    return output


# Draw edges by connecting the keypoints.
def connect_keypoints(opt, pts, edge_lists, size, basic_point_only, remove_face_labels):
    pose_pts, face_pts, hand_pts_l, hand_pts_r = pts
    w, h = size
    body_edges = np.zeros((h, w, 3), np.uint8)
    pose_edge_list, pose_color_list, hand_edge_list, hand_color_list, face_list = edge_lists

    ### pose    
    h = int(pose_pts[:, 1].max() - pose_pts[:, 1].min())
    bw = random.randrange(2, 5) if opt.isTrain else min(max(1, h // 150), 5)
    for i, edge in enumerate(pose_edge_list):
        x, y = pose_pts[edge, 0], pose_pts[edge, 1]
        if (0 not in x):
            curve_x, curve_y = interp_points(x, y)
            draw_edge(body_edges, curve_x, curve_y, bw=bw, color=pose_color_list[i],
                      draw_end_points=True)

    if not basic_point_only:
        ### hand   
        bw = random.randrange(1, 3) if opt.isTrain else min(max(1, h // 450), 3)
        for hand_pts in [hand_pts_l, hand_pts_r]:  # for left and right hand
            for i, edge in enumerate(hand_edge_list):  # for each finger
                for j in range(0, len(edge) - 1):  # for each part of the finger
                    sub_edge = edge[j:j + 2]
                    x, y = hand_pts[sub_edge, 0], hand_pts[sub_edge, 1]
                    if 0 not in x:
                        line_x, line_y = interp_points(x, y)
                        draw_edge(body_edges, line_x, line_y, bw=bw,
                                  color=hand_color_list[i], draw_end_points=False)

        ### face
        edge_len = 2
        bw = random.randrange(1, 3) if opt.isTrain else min(max(1, h // 450), 3)
        if not remove_face_labels:
            for edge_list in face_list:
                for edge in edge_list:
                    for i in range(0, max(1, len(edge) - 1), edge_len - 1):
                        sub_edge = edge[i:i + edge_len]
                        x, y = face_pts[sub_edge, 0], face_pts[sub_edge, 1]
                        if 0 not in x:
                            try:
                                curve_x, curve_y = interp_points(x, y)
                                draw_edge(body_edges, curve_x, curve_y, bw=bw, draw_end_points=False)
                            except RuntimeError:
                                continue

    return body_edges


# Normalize face keypoints according to the reference keypoints.
def normalize_faces(all_keypoints, keypoints_ref, face_ratio):
    central_keypoints = [8]
    face_centers = [np.mean(keypoints[central_keypoints, :], axis=0) for keypoints in all_keypoints]
    all_keypoints = [keypoints for keypoints, face_center in
                     zip(all_keypoints, face_centers) if face_center[0] != 0]
    face_centers = [face_center for face_center in face_centers if face_center[0] != 0]
    if len(all_keypoints) == 0: return

    part_list = [[0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10], [7, 9, 8],  # face (17)
                 [17, 26], [18, 25], [19, 24], [20, 23], [21, 22],  # eyebrows (10)
                 [27], [28], [29], [30], [31, 35], [32, 34], [33],  # nose (9)
                 [36, 45], [37, 44], [38, 43], [39, 42], [40, 47], [41, 46],  # eyes (12)
                 [48, 54], [49, 53], [50, 52], [51], [55, 59], [56, 58], [57],  # mouth (12)
                 ]

    if face_ratio is None:
        ref_dist_x, ref_dist_y = [None] * 60, [None] * 60
        dist_scale_x, dist_scale_y = [None] * 60, [None] * 60
        # print('initializing face ratio')

        valid = (keypoints_ref[:, 0] != 0) & (all_keypoints[0][:, 0] != 0)
        if len(keypoints_ref[valid, 0]) == 0:
            return
        ref_img_scale = keypoints_ref[valid, 0].max() - keypoints_ref[valid, 0].min()
        img_scale = ref_img_scale / (all_keypoints[0][valid, 0].max() - all_keypoints[0][valid, 0].min())
    else:
        dist_scale_x, dist_scale_y = face_ratio

    pts_diff = [0] * len(all_keypoints)
    for i, pts_idx in enumerate(part_list):
        if face_ratio is None:
            ### reference
            mean_dists_x, mean_dists_y = [], []
            pts = keypoints_ref[pts_idx]
            pts_cen = np.mean(pts, axis=0)
            face_cen = np.mean(keypoints_ref[central_keypoints, :], axis=0)
            for p, pt in enumerate(pts):
                mean_dists_x.append(np.linalg.norm(pt - pts_cen))
                mean_dists_y.append(np.linalg.norm(pts_cen - face_cen))
            ref_dist_x[i] = sum(mean_dists_x) / len(mean_dists_x) + 1e-3
            ref_dist_y[i] = sum(mean_dists_y) / len(mean_dists_y) + 1e-3

            ### main image
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

            dist_scale_x[i] = ref_dist_x[i] / mean_dist_x / img_scale
            dist_scale_y[i] = ref_dist_y[i] / mean_dist_y / img_scale

        for k, keypoints in enumerate(all_keypoints):
            if (keypoints[pts_idx] != 0).all():
                pts_idx_k = pts_idx
                pts = keypoints[pts_idx_k]
                face_cen = face_centers[k]
                pts_cen = np.mean(pts, axis=0)

                if 28 in pts_idx_k:
                    pts_ori = pts
                pts = (pts - pts_cen) * dist_scale_x[i] + (pts_cen - face_cen) \
                      * dist_scale_y[i] + face_cen
                if 28 in pts_idx_k:
                    pts_diff[k] = np.mean(pts_ori - pts, axis=0)
                all_keypoints[k][pts_idx_k] = pts
            else:
                all_keypoints[k][pts_idx] = 0

    for k in range(len(all_keypoints)):
        valid = all_keypoints[k][:, 0] != 0
        all_keypoints[k][valid] = all_keypoints[k][valid] + pts_diff[k]

    return [dist_scale_x, dist_scale_y]


# Define the list of keypoints that should be connected to form the edges.
def define_edge_lists(basic_point_only):
    ### pose
    pose_edge_list = [
        [17, 15], [15, 0], [0, 16], [16, 18],  # head
        [0, 1], [1, 8],  # body
        [1, 2], [2, 3], [3, 4],  # right arm
        [1, 5], [5, 6], [6, 7],  # left arm
        [8, 9], [9, 10], [10, 11],  # right leg
        [8, 12], [12, 13], [13, 14]  # left leg
    ]
    pose_color_list = [
        [153, 0, 153], [153, 0, 102], [102, 0, 153], [51, 0, 153],
        [153, 0, 51], [153, 0, 0],
        [153, 51, 0], [153, 102, 0], [153, 153, 0],
        [102, 153, 0], [51, 153, 0], [0, 153, 0],
        [0, 153, 51], [0, 153, 102], [0, 153, 153],
        [0, 102, 153], [0, 51, 153], [0, 0, 153],
    ]

    if not basic_point_only:
        pose_edge_list += [
            [11, 24], [11, 22], [22, 23],  # right foot
            [14, 21], [14, 19], [19, 20]  # left foot
        ]
        pose_color_list += [
            [0, 153, 153], [0, 153, 153], [0, 153, 153],
            [0, 0, 153], [0, 0, 153], [0, 0, 153]
        ]

    ### hand
    hand_edge_list = [
        [0, 1, 2, 3, 4],
        [0, 5, 6, 7, 8],
        [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16],
        [0, 17, 18, 19, 20]
    ]
    hand_color_list = [
        [204, 0, 0], [163, 204, 0], [0, 204, 82], [0, 82, 204], [163, 0, 204]
    ]

    ### face        
    face_list = [
        [range(0, 17)],
        [range(17, 22)],  # left eyebrow
        [range(22, 27)],  # right eyebrow
        [[28, 31], range(31, 36), [35, 28]],  # nose
        [[36, 37, 38, 39], [39, 40, 41, 36]],  # left eye
        [[42, 43, 44, 45], [45, 46, 47, 42]],  # right eye
        [range(48, 55), [54, 55, 56, 57, 58, 59, 48]],  # mouth
    ]

    return pose_edge_list, pose_color_list, hand_edge_list, hand_color_list, face_list


def func(x, a, b, c):
    return a * x ** 2 + b * x + c


def linear(x, a, b):
    return a * x + b


# Set a pixel to the given color.
# nhm modified
def set_color(im, yy, xx, color):
    if len(im.shape) == 3:
        im[yy, xx, 0], im[yy, xx, 1], im[yy, xx, 2] = color[0], color[1], color[2]
    else:
        im[yy, xx] = color[0]


# Set colors given a list of x and y coordinates for the edge.
def draw_edge(im, x, y, bw=1, color=(255, 255, 255), draw_end_points=False):
    if x is not None and x.size:
        h, w = im.shape[0], im.shape[1]
        # edge
        for i in range(-bw, bw):
            for j in range(-bw, bw):
                yy = np.maximum(0, np.minimum(h - 1, y + i))
                xx = np.maximum(0, np.minimum(w - 1, x + j))
                set_color(im, yy, xx, color)

        # edge endpoints
        if draw_end_points:
            for i in range(-bw * 2, bw * 2):
                for j in range(-bw * 2, bw * 2):
                    if (i ** 2) + (j ** 2) < (4 * bw ** 2):
                        yy = np.maximum(0, np.minimum(h - 1, np.array([y[0], y[-1]]) + i))
                        xx = np.maximum(0, np.minimum(w - 1, np.array([x[0], x[-1]]) + j))
                        set_color(im, yy, xx, color)


# Given the start and end points, interpolate to get a line.
def interp_points(x, y):
    if abs(x[:-1] - x[1:]).max() < abs(y[:-1] - y[1:]).max():
        curve_y, curve_x = interp_points(y, x)
        if curve_y is None:
            return None, None
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if len(x) < 3:
                popt, _ = curve_fit(linear, x, y)
            else:
                popt, _ = curve_fit(func, x, y)
                if abs(popt[0]) > 1:
                    return None, None
        if x[0] > x[-1]:
            x = list(reversed(x))
            y = list(reversed(y))
        curve_x = np.linspace(x[0], x[-1], math.ceil(x[-1] - x[0]))
        if len(x) < 3:
            curve_y = linear(curve_x, *popt)
        else:
            curve_y = func(curve_x, *popt)
    return curve_x.astype(int), curve_y.astype(int)
