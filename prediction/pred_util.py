import torch
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter
import recording.util as util
import argparse
import datetime
import os
import yaml
import argparse
from types import SimpleNamespace
import glob
import cv2

CONFIG_BASE_PATH = './config'


def scalar_to_classes(scalar, thresholds):
    """
    Bins a float scalar into integer class indices. Could be faster, but is hopefully readable!
    :param scalar: any shape, pytorch or numpy
    :param thresholds: list of thresholds. must be ascending
    :return:
    """
    if torch.is_tensor(scalar):
        # out = -torch.ones_like(scalar, dtype=torch.int64)
        out = torch.zeros_like(scalar, dtype=torch.int64)
    else:
        # out = -np.ones_like(scalar, dtype=np.int64)
        out = np.zeros_like(scalar, dtype=np.int64)

    for idx, threshold in enumerate(thresholds):
        out[scalar >= threshold] = idx  # may overwrite the same value many times

    # if out.min() < 0:
    #     raise ValueError('Thresholds were not broad enough')

    return out


def classes_to_scalar(classes, thresholds):
    """
    Converts an integer class array into floating values. Obviously some discretization loss here
    :param classes: any shape, pytorch or numpy
    :param thresholds: list of thresholds. must be ascending
    :param final_value: if greater than the last threshold, fill in with this value
    :return:
    """
    if torch.is_tensor(classes):    # fill with negative ones
        out = -torch.ones_like(classes, dtype=torch.float)
    else:
        out = -np.ones_like(classes, dtype=np.float)

    for idx, threshold in enumerate(thresholds):
        if idx == 0:
            val = thresholds[0]
        elif idx == len(thresholds) - 1:
            final_value = thresholds[-1] + (thresholds[-1] - thresholds[-2]) / 2    # Set it equal to the last value, plus half to gap to the previous thresh
            val = final_value
        else:
            val = (thresholds[idx] + thresholds[idx + 1]) / 2

        out[classes == idx] = val

    if out.min() < 0:
        raise ValueError('Thresholds were not broad enough')

    return out


def contactmask_to_scalar(classes_pred, force_thresholds):
    force_pred_class = torch.argmax(classes_pred[:, 1:, :, :], dim=1) + 1   # Skip the first contact class
    force_pred_scalar = classes_to_scalar(force_pred_class, force_thresholds)

    input_contact = classes_pred[:, 0, :, :] < 0

    force_pred_scalar[input_contact] = 0

    return force_pred_scalar


def parse_config_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--config', type=str, required=True)
    args = parser.parse_args()
    return load_config(args.config)


def load_config(config_name):
    config_path = os.path.join(CONFIG_BASE_PATH, config_name + '.yml')
    print('Loading config file:', config_path)

    with open(config_path, 'r') as stream:
        data = yaml.safe_load(stream)

    # data_obj = namedtuple('MyTuple', data)
    data_obj = SimpleNamespace(**data)
    data_obj.CONFIG_NAME = config_name
    return data_obj


def find_latest_checkpoint(config_name):
    """
    Finds the newest model checkpoint file, sorted by the date of the file
    """
    all_checkpoints = glob.glob('data/model/*.pth')
    possible_matches = []
    for p in all_checkpoints:
        f = os.path.basename(p)
        if not f.startswith(config_name):
            continue
        f = f[len(config_name):-4] # cut off the prefix and suffix
        if not f.lower().islower():     # if it has any letters
            possible_matches.append(p)

    if len(possible_matches) == 0:
        raise ValueError('No valid model checkpoint files found')

    latest_file = max(possible_matches, key=os.path.getctime)
    print('Loading checkpoint file:', latest_file)

    return latest_file


def resnet_preprocessor(rgb):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    if rgb.shape[2] == 12:
        mean = mean.repeat(4)
        std = std.repeat(4)

    rgb = rgb - mean
    rgb = rgb / std
    return rgb


def resnet_invert_preprocess(rgb):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # rgb = rgb - mean
    rgb = rgb * std + mean
    return rgb


def run_model(img, model, config):
    with torch.no_grad():
        model_output = model(img.cuda())
        force_pred_class = model_output[0]

        force_pred_class = torch.argmax(force_pred_class, dim=1)
        force_pred_scalar = classes_to_scalar(force_pred_class, config.FORCE_THRESHOLDS)

    return force_pred_scalar.detach()


def process_and_run_model(img, best_model, config):
    # Takes in a cropped image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255
    img = resnet_preprocessor(img)
    img = img.transpose(2, 0, 1).astype('float32')
    img = torch.tensor(img).unsqueeze(0)

    force_pred_scalar = run_model(img, best_model, config)

    force_pred_scalar = force_pred_scalar.detach().cpu().squeeze().numpy()

    return force_pred_scalar


def process_and_run_batched_model(images, best_model, config):
    # Takes in a cropped image
    image_shape = images[0].shape
    images_tensor = torch.zeros((len(images), image_shape[2], image_shape[0], image_shape[1]), dtype=torch.float32)

    for idx, img in enumerate(images):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float32') / 255
        # img = seq_reader.crop_img(img, 0, config)
        img = resnet_preprocessor(img)
        img = img.transpose(2, 0, 1)
        images_tensor[idx, :, :, :] = torch.tensor(img)

    force_pred_scalar = run_model(images_tensor, best_model, config)
    force_pred_scalar = force_pred_scalar.detach().cpu().numpy()

    output_pressure = []
    for i in range(len(images)):
        output_pressure.append(force_pred_scalar[i, :, :])

    # force_color_pred = pressure_to_colormap(force_pred_scalar)
    return output_pressure


def get_hand_bbox(img, mp_wrapper):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = mp_wrapper.run_img(img, vis=False)
    scale = 1.5

    hand_joints = result['right_points']
    if result['left_points'] is not None:
        hand_joints = result['left_points']

    if hand_joints is not None:
        hand_joints = np.array(hand_joints)
        hand_joints[:, 0] *= img.shape[1]
        hand_joints[:, 1] *= img.shape[0]

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
    out_dict['max_x'] = min(img.shape[1], center_x + radius)
    out_dict['min_y'] = max(0, center_y - radius)
    out_dict['max_y'] = min(img.shape[0], center_y + radius)

    for key, value in out_dict.items():
        out_dict[key] = int(round(value))

    return out_dict


def get_two_hands_bbox(img, mp_wrapper):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = mp_wrapper.run_img(img, vis=False)
    scale = 1.5

    hand_bboxes = list()

    for hand_joints in [result['right_points'], result['left_points']]:
        if hand_joints is None:
            continue

        hand_joints = np.array(hand_joints)
        hand_joints[:, 0] *= img.shape[1]
        hand_joints[:, 1] *= img.shape[0]

        center_x = (hand_joints[:, 0].min() + hand_joints[:, 0].max()) / 2
        center_y = (hand_joints[:, 1].min() + hand_joints[:, 1].max()) / 2

        radius = max(hand_joints[:, 0].max() - center_x, hand_joints[:, 1].max() - center_y)
        radius = radius * scale

        out_dict = dict()
        out_dict['min_x'] = max(0, center_x - radius)
        out_dict['max_x'] = min(img.shape[1], center_x + radius)
        out_dict['min_y'] = max(0, center_y - radius)
        out_dict['max_y'] = min(img.shape[0], center_y + radius)

        for key, value in out_dict.items():
            out_dict[key] = int(round(value))

        hand_bboxes.append(out_dict)

    return hand_bboxes


def draw_bbox_full_frame(img, pose_bbox, half=False):
    if half:
        max_x = int(round(pose_bbox['max_x'] / 2))
        min_x = int(round(pose_bbox['min_x'] / 2))
        max_y = int(round(pose_bbox['max_y'] / 2))
        min_y = int(round(pose_bbox['min_y'] / 2))
    else:
        max_x = int(round(pose_bbox['max_x']))
        min_x = int(round(pose_bbox['min_x']))
        max_y = int(round(pose_bbox['max_y']))
        min_y = int(round(pose_bbox['min_y']))

    cv2.line(img, (min_x, min_y), (max_x, min_y), (0, 0, 255), 4)
    cv2.line(img, (min_x, max_y), (max_x, max_y), (0, 0, 255), 4)
    cv2.line(img, (min_x, min_y), (min_x, max_y), (0, 0, 255), 4)
    cv2.line(img, (max_x, min_y), (max_x, max_y), (0, 0, 255), 4)


def draw_finger_bar_graph(img, finger, start_pos, color):
    for i in range(len(finger)):
        finger_val = finger[i]
        start_x = start_pos + i * 50
        end_x = start_x + 20
        start_y = 300
        end_y = start_y - int(finger_val * 200)

        cv2.rectangle(img, (start_x, start_y), (end_x, end_y), color, -1)


def get_full_frame_pressure(cropped_pressure_img, pose_bbox):
    full_frame_pressure = np.zeros((1080, 1920, 3), dtype=np.uint8)
    max_x = int(round(float(pose_bbox['max_x'])))
    min_x = int(round(float(pose_bbox['min_x'])))
    max_y = int(round(float(pose_bbox['max_y'])))
    min_y = int(round(float(pose_bbox['min_y'])))

    resize_dims = (max_x - min_x, max_y-min_y)
    resized_pressure = cv2.resize(cropped_pressure_img, resize_dims)
    full_frame_pressure[min_y:max_y, min_x:max_x, :] = resized_pressure
    return full_frame_pressure


def get_full_frame_1d(cropped_pressure_img, pose_bbox, destination_size=None, full_frame_pressure=None):
    if full_frame_pressure is None:
        if destination_size is None:
            full_frame_pressure = np.zeros((1080, 1920))
        else:
            full_frame_pressure = np.zeros((destination_size[0], destination_size[1]))

    max_x = int(round(pose_bbox['max_x']))
    min_x = int(round(pose_bbox['min_x']))
    max_y = int(round(pose_bbox['max_y']))
    min_y = int(round(pose_bbox['min_y']))

    resize_dims = (max_x - min_x, max_y-min_y)
    resized_pressure = cv2.resize(cropped_pressure_img, resize_dims)
    full_frame_pressure[min_y:max_y, min_x:max_x] += resized_pressure
    return full_frame_pressure


def opencv_draw_rect(img, min_x, min_y, max_x, max_y, color=(0, 0, 255), thickness=2):
    min_x = int(round(min_x))
    min_y = int(round(min_y))
    max_x = int(round(max_x))
    max_y = int(round(max_y))

    cv2.line(img, (min_x, min_y), (max_x, min_y), color, thickness)
    cv2.line(img, (min_x, min_y), (min_x, max_y), color, thickness)
    cv2.line(img, (max_x, min_y), (max_x, max_y), color, thickness)
    cv2.line(img, (min_x, max_y), (max_x, max_y), color, thickness)

