import cv2
import glob
import numpy as np
from torch.utils.data import DataLoader, Dataset
import albumentations as albu
from recording.sequence_reader import SequenceReader
import random
from prediction.pred_util import scalar_to_classes


def get_training_augmentation(method):
    if method == 0:
        train_transform = [
            albu.HorizontalFlip(p=0.5),
            albu.ColorJitter(brightness=0.3, contrast=0.3, p=0.5),
            albu.ShiftScaleRotate(rotate_limit=10)
        ]
    elif method == 1:
        train_transform = [
            albu.HorizontalFlip(p=0.5),
            albu.ShiftScaleRotate(rotate_limit=10, p=0.2),
            albu.Blur(p=0.2),
            albu.ColorJitter(brightness=0.3, contrast=0.3, p=0.3),
            albu.HueSaturationValue(p=0.2),
            albu.RandomBrightnessContrast(p=0.2),
            albu.Sharpen(p=0.2)
        ]

    return albu.Compose(train_transform)


class ForceDataset(Dataset):
    def __init__(
            self,
            config,
            seq_filter,
            preprocessing_fn=None,
            image_method=0,
            force_method=False,
            skip_frames=1,
            randomize_cam_seq=True,
            phase='val'
    ):
        self.config = config
        self.skip_frames = skip_frames
        self.randomize_cam_seq = randomize_cam_seq
        self.phase = phase

        self.per_seq_datapoints = None
        if hasattr(config, 'USE_CAMERAS'):
            self.all_datapoints = self.load_sequences(seq_filter, config.USE_CAMERAS)
        else:
            self.all_datapoints = self.load_sequences(seq_filter)

        self.include_full_frame = False
        if hasattr(config, 'INCLUDE_FULL_FRAME'):
            self.include_full_frame = True

        self.augmentor = None
        if config.DO_AUG and phase == 'train':
            self.augmentor = get_training_augmentation(method=config.AUG_METHOD)

        if len(self.all_datapoints) == 0:
            raise ValueError('Couldnt find datapoints')

        self.image_method = image_method
        self.force_method = force_method
        self.preprocessing_fn = preprocessing_fn

        print('Loaded dataset with filter: {}. Frame subsampling: {}. Frames loaded: {}'.format(seq_filter, skip_frames, len(self.all_datapoints)))

    def __getitem__(self, i):
        timestep = self.all_datapoints[i]['timestep']
        camera_idx = self.all_datapoints[i]['camera_idx']
        seq_reader = self.all_datapoints[i]['seq_reader']

        network_image_size = (self.config.NETWORK_IMAGE_SIZE_X, self.config.NETWORK_IMAGE_SIZE_Y)
        force_array = seq_reader.get_force_pytorch(camera_idx, timestep, self.config)
        image_0 = seq_reader.get_img_pytorch(camera_idx, timestep, self.config)
        image_out = image_0

        force_array = cv2.resize(force_array, network_image_size)
        raw_force_array = force_array

        force_array = scalar_to_classes(force_array, self.config.FORCE_THRESHOLDS)

        if self.augmentor is not None:
            augmented = self.augmentor(image=image_out, mask=force_array)
            image_out = augmented['image']
            force_array = augmented['mask'].astype(np.int64)

        if self.preprocessing_fn is not None:
            image_out = self.preprocessing_fn(image_out)

        fingers_data_list = seq_reader.fingers_data
        if self.config.WEAK_LABEL_HIGH_LOW:
            fingers_data_list = list(fingers_data_list)
            if 'onebyone' in seq_reader.action:
                fingers_data_list.append(-1)
            elif 'low' in seq_reader.action:
                fingers_data_list.append(0)
            elif 'high' in seq_reader.action:
                fingers_data_list.append(1)
            else:
                fingers_data_list.append(-1)

        fingers_data = np.array(fingers_data_list, dtype=np.float32)
        if raw_force_array.sum() < 500 and not seq_reader.weak_label:   # For fully labeled data, set the contact label to zero when theres no force
            fingers_data *= 0

        out_dict = dict()
        out_dict['img'] = self.to_tensor(image_out)
        out_dict['img_original'] = self.to_tensor(image_0)
        out_dict['force'] = force_array
        out_dict['seq_path'] = seq_reader.seq_path
        out_dict['camera_idx'] = camera_idx
        out_dict['timestep'] = timestep
        out_dict['participant'] = seq_reader.participant
        out_dict['action'] = seq_reader.action
        out_dict['fingers'] = fingers_data
        out_dict['raw_force'] = raw_force_array

        if self.include_full_frame:
            out_dict['img_full_frame'] = self.to_tensor(seq_reader.get_img_pytorch(camera_idx, timestep, self.config, nocrop=True))
            out_dict['pose_bbox'] = seq_reader.get_pose_bbox(camera_idx, timestep)

        return out_dict

    def __len__(self):
        return len(self.all_datapoints)

    def to_tensor(self, x):
        if len(x.shape) == 3:   # image
            return x.transpose(2, 0, 1).astype('float32')
        elif len(x.shape) == 4:   # video
            return x.astype('float32')
        else:
            raise ValueError('Wrong number of channels')

    def load_sequences(self, seq_filter, use_cameras=None):
        if not isinstance(seq_filter, list):
            raise ValueError('Need a sequence filter list!')

        datapoints = []

        all_sequences = []
        for filter in seq_filter:
            all_sequences.extend(glob.glob(filter))

        for seq_path in all_sequences:
            if any([exclude in seq_path for exclude in self.config.EXCLUDE_ACTIONS]):
                continue

            seq_reader = SequenceReader(seq_path)
            for c in seq_reader.camera_ids:
                if use_cameras is not None and c not in use_cameras:
                    continue

                this_camera_points = []
                for t in range(seq_reader.num_frames):
                    if self.config.SKIP_FRAMES_WITHOUT_POSE:
                        if seq_reader.get_hand_joints(c, t) is None:    # Didn't get a hand pose at this timestep, ignore
                            continue

                    if not seq_reader.weak_label:
                        if seq_reader.sensel_homography[c][t] is None or seq_reader.sensel_homography[c][t].size <= 1:
                            continue    # Throw out frames with no sensel homography

                    datapoint = dict()
                    datapoint['seq_reader'] = seq_reader
                    datapoint['camera_idx'] = c
                    datapoint['timestep'] = t
                    this_camera_points.append(datapoint)
                datapoints.append(this_camera_points)

        if self.randomize_cam_seq:
            random.shuffle(datapoints)

        self.per_seq_datapoints = datapoints
        flattened_preskip = [item for sublist in datapoints for item in sublist]
        flattened = flattened_preskip[::self.skip_frames]
        # print('Before skip length {}, after skip length {}'.format(len(flattened_preskip), len(flattened)))

        return flattened

