import cv2
import argparse
import torch
import numpy as np
from prediction.pred_util import find_latest_checkpoint, get_hand_bbox, get_full_frame_1d, draw_bbox_full_frame, process_and_run_model, load_config
from recording.util import set_subframe, pressure_to_colormap
from pose.mediapipe_minimal import MediaPipeWrapper

disp_x = 1920
disp_y = 1080


def webcam_demo():
    window_name = 'PressureVision++ Webcam Demo'
    disp_frame = np.zeros((disp_y, disp_x, 3), dtype=np.uint8)
    mp_wrapper = MediaPipeWrapper()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, 30)

    # If your webcam has variable focus, often it is better to turn it off
    # cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Turn the autofocus off
    # cap.set(cv2.CAP_PROP_FOCUS, 15)  # Set to predetermined focus value for BRIO

    best_model = torch.load(find_latest_checkpoint(config.CONFIG_NAME))
    best_model.eval()

    while True:
        ret, camera_frame = cap.read()
        if camera_frame is None:
            continue

        base_img = camera_frame

        bbox = get_hand_bbox(base_img, mp_wrapper)
        crop_frame = base_img[bbox['min_y']:bbox['max_y'], bbox['min_x']:bbox['max_x'], ...]
        crop_frame = cv2.resize(crop_frame, (config.NETWORK_IMAGE_SIZE_X, config.NETWORK_IMAGE_SIZE_Y))    # image is YX

        force_pred = process_and_run_model(crop_frame, best_model, config)
        force_pred_full_frame = get_full_frame_1d(force_pred, bbox)
        force_pred_color_full_frame = pressure_to_colormap(force_pred_full_frame)

        overlay_frame = cv2.addWeighted(base_img, 0.6, force_pred_color_full_frame, 1.0, 0.0)
        draw_bbox_full_frame(overlay_frame, bbox)

        set_subframe(0, base_img, disp_frame, title='Raw Camera Frame')
        set_subframe(1, crop_frame, disp_frame, title='Network Input')
        set_subframe(2, overlay_frame, disp_frame, title='Network Output with Overlay')
        set_subframe(3, pressure_to_colormap(force_pred_full_frame), disp_frame, title='Network Output')

        cv2.imshow(window_name, disp_frame)
        keycode = cv2.waitKey(1) & 0xFF

        if keycode == ord('q'):
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--config', type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    webcam_demo()
