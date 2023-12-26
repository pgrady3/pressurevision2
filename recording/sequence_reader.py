import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import argparse
import glob
import random
import os
from recording.util import *
import json
import yaml


action_list_paths = ['config/action_lists/weak_label_actions.yml']
action_list = None


def get_fingers_data(query_action):
    query_action = query_action.replace(' ', '_')

    global action_list
    if action_list is None:
        action_list = dict()
        for a_path in action_list_paths:
            with open(a_path, 'r') as stream:
                a_data = yaml.safe_load(stream)
                action_list.update(a_data)

    for a in action_list.keys():
        if query_action.startswith(a):
            l = list(action_list[a])
            return l

    raise ValueError('Dont know weak label for action: ' + query_action)


class SequenceReader:
    def __init__(self, seq_path):
        self.seq_path = seq_path

        split_path = os.path.normpath(seq_path).split(os.path.sep)
        self.action = split_path[-1]
        self.participant = split_path[-2]

        self.metadata = json_read(os.path.join(self.seq_path, 'meta.json'))
        # self.num_cameras = self.metadata['num_cameras']
        self.camera_ids = self.metadata['camera_ids']
        self.num_frames = self.metadata['num_frames']
        self.timesteps = self.metadata['timesteps']

        self.fingers_data = np.array(get_fingers_data(self.action), dtype=int)
        self.weak_label = False

        self.img_height = dict()
        self.img_width = dict()
        self.sensel_homography = dict()
        self.sensel_points = dict()

        if not self.metadata['is_weak']:   # Fully labeled sequence
            for camera_id in self.camera_ids:
                self.sensel_homography[camera_id] = dict()
                self.sensel_points[camera_id] = dict()
                for t in range(len(self.timesteps)):
                    self.sensel_homography[camera_id][t] = np.array(self.metadata['camera_calibrations'][camera_id][str(t)]['homography'])
                    self.sensel_points[camera_id][t] = np.array(self.metadata['camera_calibrations'][camera_id][str(t)]['imgpts'])

                self.img_width[camera_id] = 1920
                self.img_height[camera_id] = 1080

        else:   # Weakly labeled sequence
            self.weak_label = True
            for camera_id in self.camera_ids:
                self.img_width[camera_id] = 1920
                self.img_height[camera_id] = 1080

        self.pose_estimate_data = self.load_pose_estimates()

    def load_pose_estimates(self):
        pose_path = os.path.join('data', 'pose_estimates', self.participant, self.action + '.pkl')
        if not os.path.exists(pose_path):
            return None

        data = pkl_read(pose_path)
        for c_id in self.camera_ids:
            for f_key, d in data[c_id].items():
                if d['right_points'] is not None:
                    d['right_points'] = np.array(d['right_points'])
                    d['right_points'][:, 0] *= self.img_width[c_id]
                    d['right_points'][:, 1] *= self.img_height[c_id]
                    d['right_points'] = d['right_points'].round().astype(int)
                if d['left_points'] is not None:
                    d['left_points'] = np.array(d['left_points'])
                    d['left_points'][:, 0] *= self.img_width[c_id]
                    d['left_points'][:, 1] *= self.img_height[c_id]
                    d['left_points'] = d['left_points'].round().astype(int)

        return data

    def get_hand_joints(self, camera_idx, frame_idx):
        d = self.pose_estimate_data[camera_idx][frame_idx]

        if d['right_points'] is None and d['left_points'] is None:
            return None

        if d['right_points'] is None:
            return d['left_points']
        else:
            if d['left_points'] is None:
                return d['right_points']
            else:
                # In cases where two hands are detected, perform a weighting scheme to decide which one we choose
                # The scheme factors which bounding box is larger, and which is closer to the center of the camera

                spread_right_x = d['right_points'][:, 1].max() - d['right_points'][:, 1].min()
                spread_right_y = d['right_points'][:, 0].max() - d['right_points'][:, 0].min()
                spread_left_x = d['left_points'][:, 1].max() - d['left_points'][:, 1].min()
                spread_left_y = d['left_points'][:, 0].max() - d['left_points'][:, 0].min()

                disp_right_x = d['right_points'][:, 1].mean() - self.img_height[camera_idx] / 2
                disp_right_y = d['right_points'][:, 0].mean() - self.img_width[camera_idx] / 2
                disp_right = np.linalg.norm([disp_right_x, disp_right_y])

                disp_left_x = d['left_points'][:, 1].mean() - self.img_height[camera_idx] / 2
                disp_left_y = d['left_points'][:, 0].mean() - self.img_width[camera_idx] / 2
                disp_left = np.linalg.norm([disp_left_x, disp_left_y])

                size_r_over_l = spread_right_x + spread_right_y - (spread_left_x + spread_left_y)
                disp_r_over_l = disp_right - disp_left

                score = size_r_over_l * 3 - disp_r_over_l

                # print('Two hands. Right {} left {}'.format(spread_right, spread_left))
                if score > 0:
                    return d['right_points']
                else:
                    return d['left_points']

    def get_pose_bbox(self, camera_idx, frame_idx, scale=1.5):
        hand_joints = self.get_hand_joints(camera_idx, frame_idx)
        if hand_joints is not None:
            center_x = (hand_joints[:, 0].min() + hand_joints[:, 0].max()) / 2
            center_y = (hand_joints[:, 1].min() + hand_joints[:, 1].max()) / 2

            radius = max(hand_joints[:, 0].max() - center_x, hand_joints[:, 1].max() - center_y)
            radius = radius * scale
        else:
            center_x = 960
            center_y = 540
            radius = 10000

        out_dict = dict()
        out_dict['min_x'] = max(0, center_x - radius)
        out_dict['max_x'] = min(self.img_width[camera_idx], center_x + radius)
        out_dict['min_y'] = max(0, center_y - radius)
        out_dict['max_y'] = min(self.img_height[camera_idx], center_y + radius)

        for key, value in out_dict.items():
            out_dict[key] = int(round(value))

        return out_dict

    def get_hand_crop(self, img, camera_idx, frame_idx, config):
        bbox = self.get_pose_bbox(camera_idx, frame_idx)
        img = img[bbox['min_y']:bbox['max_y'], bbox['min_x']:bbox['max_x'], ...]

        network_image_size = (config.NETWORK_IMAGE_SIZE_X, config.NETWORK_IMAGE_SIZE_Y)
        out_img = cv2.resize(img, network_image_size)    # image is YX
        return out_img

    def get_img(self, camera_idx, frame_idx, to_rgb=False):
        img = cv2.imread(self.get_img_path(camera_idx, frame_idx))
        if to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def get_force_pytorch(self, camera_idx, frame_idx, config, override_force_path=None):
        force = self.get_force_warped_to_img(camera_idx, frame_idx, override_force_path=override_force_path).astype('float32')

        if config.HAND_CROP is True:
            force = self.get_hand_crop(force, camera_idx, frame_idx, config)

        return force

    def get_img_pytorch(self, camera_idx, frame_idx, config, nocrop=False):
        img = self.get_img(camera_idx, frame_idx, to_rgb=True).astype('float32') / 255

        if config.HAND_CROP is True and not nocrop:
            img = self.get_hand_crop(img, camera_idx, frame_idx, config)

        return img

    def get_img_path(self, camera_idx, frame_idx):
        return os.path.join(self.seq_path, 'camera_{}'.format(camera_idx), '{:05d}.jpg'.format(frame_idx))

    def get_pressure_kPa(self, frame_idx):
        if self.weak_label:
            # print('Trying to get pressure from a weakly labeled sequence', self.seq_path)
            return np.zeros((105, 185))

        pkl_path = os.path.join(self.seq_path, 'force', '{:05d}.pkl'.format(frame_idx))
        with open(pkl_path, 'rb') as handle:
            raw_counts = pickle.load(handle)

        kPa = convert_counts_to_kPa(raw_counts)
        return kPa

    def get_force_warped_to_img(self, camera_idx, frame_idx, draw_sensel=False, override_force_path=None):
        if self.weak_label:
            # print('Trying to get pressure from a weakly labeled sequence', self.seq_path)
            return np.zeros((self.img_height[camera_idx], self.img_width[camera_idx]))

        if override_force_path is None:
            force_img = self.get_pressure_kPa(frame_idx)
        else:
            save_path = os.path.join(override_force_path, self.participant, self.action, '{:05d}.pkl'.format(frame_idx))
            force_img = pkl_read(save_path)

        if self.sensel_homography[camera_idx][frame_idx] is None or self.sensel_homography[camera_idx][frame_idx].size <= 1:
            print('Tried to get force from failed homogrophy', camera_idx, frame_idx, self.seq_path)
            return np.zeros((self.img_height[camera_idx], self.img_width[camera_idx]))

        force_warped = cv2.warpPerspective(force_img, self.sensel_homography[camera_idx][frame_idx], (self.img_width[camera_idx], self.img_height[camera_idx]))

        if draw_sensel:
            for c_idx in range(4):  # Draw the four corners on the image
                start_point = tuple(self.sensel_points[camera_idx][frame_idx][c_idx, :].astype(int))
                end_point = tuple(self.sensel_points[camera_idx][frame_idx][(c_idx + 1) % 4, :].astype(int))
                cv2.line(force_warped, start_point, end_point, 20, 3)

        return force_warped

    def get_force_overlay_img(self, camera_idx, frame_idx, draw_sensel=False):
        force_warped = self.get_force_warped_to_img(camera_idx, frame_idx, draw_sensel=draw_sensel)
        force_color_warped = pressure_to_colormap(force_warped, colormap=cv2.COLORMAP_OCEAN)
        img = self.get_img(camera_idx, frame_idx)

        return cv2.addWeighted(img, 1.0, force_color_warped, 1.0, 0.0)

    def get_overall_frame(self, frame_idx, overlay_force=True, draw_sensel=False):
        """
        Returns a frame with all views and cameras rendered as subwindows
        :return: A numpy array
        """
        out_x = 1920  # Rendering X, y
        out_y = 1080

        cur_frame = np.zeros((out_y, out_x, 3), dtype=np.uint8)

        force = self.get_pressure_kPa(frame_idx)
        set_subframe(0, pressure_to_colormap(force), cur_frame, steps_x=3, steps_y=3, title='Sensel')

        for idx, c in enumerate(self.camera_ids):
            if overlay_force:
                img = self.get_force_overlay_img(c, frame_idx, draw_sensel=draw_sensel)
            else:
                img = self.get_img(c, frame_idx)
            set_subframe(idx + 1, img, cur_frame, steps_x=3, steps_y=3, title=c)

        return cur_frame
