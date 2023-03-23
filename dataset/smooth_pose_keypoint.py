# this file is used to smooth pose keypoints
import os
import json_tricks as json
import numpy as np


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


def smooth_points(pts_list_npy):
    num_pt = pts_list_npy.shape[1]
    new_cur_pts_list = []
    for ii in range(num_pt):
        cur_pt_seq = pts_list_npy[:, ii, :]
        cur_pt_seq_cumsum = np.cumsum(cur_pt_seq, axis=0)
        cur_pt_valid = [int(0 not in x) for x in cur_pt_seq]
        cur_pt_cumsum_num = np.cumsum(cur_pt_valid, axis=0)
        num_frame = cur_pt_seq.shape[0]
        # win_len = 5
        new_cur_pts = np.zeros_like(cur_pt_seq)
        new_cur_pts[0] = cur_pt_seq[0]
        new_cur_pts[1] = cur_pt_seq_cumsum[2]/cur_pt_cumsum_num[2] if cur_pt_cumsum_num[2] else cur_pt_seq[1]
        new_cur_pts[2] = cur_pt_seq_cumsum[4]/cur_pt_cumsum_num[4] if cur_pt_cumsum_num[4] else cur_pt_seq[2]
        for jj in range(3, num_frame-2):
            if (cur_pt_cumsum_num[jj+2] - cur_pt_cumsum_num[jj-3]):
                new_cur_pts[jj] = (cur_pt_seq_cumsum[jj+2] - cur_pt_seq_cumsum[jj-3])\
                                  /(cur_pt_cumsum_num[jj+2] - cur_pt_cumsum_num[jj-3])
            else:
                new_cur_pts[jj] = cur_pt_seq[jj]
        if (cur_pt_cumsum_num[-1] - cur_pt_cumsum_num[-4]):
            new_cur_pts[num_frame-2] = (cur_pt_seq_cumsum[-1] - cur_pt_seq_cumsum[-4])\
                                       /(cur_pt_cumsum_num[-1] - cur_pt_cumsum_num[-4])
        else:
            new_cur_pts[num_frame-2] = cur_pt_seq[num_frame-2]
        new_cur_pts[num_frame-1] = cur_pt_seq[-1]
        # reset invalid point to be (0, 0)
        new_cur_pts[cur_pt_valid == 0] = [0, 0]
        new_cur_pts_list.append(new_cur_pts)
    new_cur_pts_list_npy = np.stack(new_cur_pts_list, axis=1)
    return new_cur_pts_list_npy


if __name__ == "__main__":
    msk_json_path = "/data/youtube-dance/output/clean/clean_unseen_video_dict.json"
    label_dir_path = "/data/youtube-dance/output/checked_openpose"
    new_label_dir_path = "/data/youtube-dance/output/smooth_openpose"
    os.makedirs(new_label_dir_path, exist_ok=True)
    edge_lists = define_edge_lists(basic_point_only=False)
    n_frame_total = 30
    with open(msk_json_path, "r") as f:
        msk_video_dict = json.load(f)
    msk_video_list = list(msk_video_dict.keys())
    new_msK_video_list = {}
    for video_name in msk_video_list:
        print(video_name)
        frame_list = msk_video_dict[video_name]
        frame_list.sort()
        msk_list = [os.path.join(label_dir_path, "%05d" % int(video_name), frame[:-4]+"_keypoints.json")
                    for frame in frame_list]
        msk_list = msk_list[:n_frame_total]
        pose_pts_list = []
        face_pts_list = []
        hand_pts_l_list = []
        hand_pts_r_list = []
        name_list = []
        for json_input in msk_list:
            name_list.append((os.path.basename(json_input)).split("_")[0])
            with open(json_input, encoding='utf-8') as f:
                keypoint_dicts = json.loads(f.read())["people"]
            keypoint_dict = keypoint_dicts[0]
            pose_pts = np.array(keypoint_dict["pose_keypoints_2d"]).reshape(25, 3)
            face_pts = np.array(keypoint_dict["face_keypoints_2d"]).reshape(70, 3)
            hand_pts_l = np.array(keypoint_dict["hand_left_keypoints_2d"]).reshape(21, 3)
            hand_pts_r = np.array(keypoint_dict["hand_right_keypoints_2d"]).reshape(21, 3)
            pts = [extract_valid_keypoints(pts, edge_lists) for pts in [pose_pts, face_pts, hand_pts_l, hand_pts_r]]
            pose_pts_list.append(pts[0])
            face_pts_list.append(pts[1])
            hand_pts_l_list.append(pts[2])
            hand_pts_r_list.append(pts[3])
        pose_pts_list_npy = np.stack(pose_pts_list, axis=0)
        face_pts_list_npy = np.stack(face_pts_list, axis=0)
        hand_pts_l_list_npy = np.stack(hand_pts_l_list, axis=0)
        hand_pts_r_list_npy = np.stack(hand_pts_r_list, axis=0)
        new_pose_pts_list_npy = smooth_points(pose_pts_list_npy)
        new_face_pts_list_npy = smooth_points(face_pts_list_npy)
        new_hand_pts_l_list_npy = smooth_points(hand_pts_l_list_npy)
        new_hand_pts_r_list_npy = smooth_points(hand_pts_r_list_npy)
        # save
        new_keypoint_dict = {}
        new_keypoint_dict["pose_keypoints_2d"] = new_pose_pts_list_npy
        new_keypoint_dict["face_keypoints_2d"] = new_face_pts_list_npy
        new_keypoint_dict["hand_left_keypoints_2d"] = new_hand_pts_l_list_npy
        new_keypoint_dict["hand_right_keypoints_2d"] = new_hand_pts_r_list_npy
        new_keypoint_dict["name"] = name_list
        new_json_name = os.path.join(new_label_dir_path, "%05d.json" % int(video_name))
        with open(new_json_name, "w") as f:
            json.dump(new_keypoint_dict, f)
