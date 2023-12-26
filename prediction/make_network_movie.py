import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import argparse
import glob
import random
import torch
import numpy as np
from prediction.loader import ForceDataset
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from prediction.model_builder import build_model
from prediction.pred_util import *
import random
import torch.multiprocessing
from recording.util import *
torch.multiprocessing.set_sharing_strategy('file_system')


def test(num_frames=8000):
    config.DATALOADER_TEST_SKIP_FRAMES = 1
    config.VAL_FILTER = config.TEST_FILTER + config.TEST_WEAK_FILTER
    config.INCLUDE_FULL_FRAME = True

    random.seed(5)  # Set the seed so the sequences will be randomized the same
    model_dict = build_model(config, device, ['val'])

    best_model = torch.load(find_latest_checkpoint(config.CONFIG_NAME))
    best_model.eval()

    val_dataloader = DataLoader(model_dict['val_dataset'], batch_size=1, shuffle=False, num_workers=config.NUM_WORKERS)

    out_path = os.path.join('data', 'movies', config.CONFIG_NAME + '_movie.avi')
    print('Saving to:', out_path)
    mw = MovieWriter(out_path, fps=15)

    for idx, batch in enumerate(tqdm(val_dataloader)):
        image_model = batch['img']
        force_gt = batch['raw_force']
        participant = batch['participant'][0]
        action = batch['action'][0]
        camera_id = batch['camera_idx'][0]

        with torch.no_grad():
            # Run the model
            model_output = best_model(image_model.cuda())
            force_pred_class = model_output[0]
            fingers_pred = model_output[1]['bottleneck_logits'].detach().squeeze().cpu().numpy()

            force_pred_class = torch.argmax(force_pred_class, dim=1)
            force_pred_scalar = classes_to_scalar(force_pred_class, config.FORCE_THRESHOLDS)

            # Render the estimated force
            pose_bbox = batch['pose_bbox']

            image_save = batch['img_full_frame'].squeeze().numpy().transpose((1, 2, 0))
            image_save = cv2.cvtColor(image_save * 255, cv2.COLOR_BGR2RGB).astype(np.uint8)
            force_color_gt = get_full_frame_pressure(pressure_to_colormap(force_gt.detach().squeeze().cpu().numpy()), pose_bbox)
            force_color_pred = get_full_frame_pressure(pressure_to_colormap(force_pred_scalar.detach().squeeze().cpu().numpy()), pose_bbox)

            opencv_draw_rect(image_save, pose_bbox['min_x'].item(), pose_bbox['min_y'].item(), pose_bbox['max_x'].item(), pose_bbox['max_y'].item(), color=(0, 0, 255), thickness=4)    # Draw crop bbox

            # Overlay the estimated force on the original image
            val_img = 0.6
            force_color_gt = cv2.addWeighted(force_color_gt, 1.0, image_save, val_img, 0)
            force_color_pred = cv2.addWeighted(force_color_pred, 1.0, image_save, val_img, 0)

            img_weak_label = np.zeros((384, 480, 3), dtype=np.uint8)

            if fingers_pred is not None:
                fingers_gt = batch['fingers'].detach().squeeze().cpu().numpy()
                draw_finger_bar_graph(img_weak_label, fingers_gt, 40, (100, 255, 100))
                draw_finger_bar_graph(img_weak_label, fingers_pred, 60, (255, 100, 100))

            out_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            set_subframe(0, image_save, out_frame, title='RGB Image')
            set_subframe(1, img_weak_label, out_frame, title='Weak Labels')
            set_subframe(2, force_color_gt, out_frame, title='GT Pressure')
            set_subframe(3, force_color_pred, out_frame, title='Est Pressure')
            cv2.putText(out_frame, '{} {}'.format(participant, batch['timestep'][0].item()), (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            cv2.putText(out_frame, '{}'.format(action), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            cv2.putText(out_frame, '{}'.format(camera_id), (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

            mw.write_frame(out_frame)

        if idx > num_frames:
            break

    mw.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=8000)
    parser.add_argument('-cfg', '--config', type=str)
    args = parser.parse_args()
    config = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test(args.frames)
