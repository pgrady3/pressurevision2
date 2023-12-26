import glob
from recording.sequence_reader import SequenceReader
import os
from pose.mediapipe_minimal import MediaPipeWrapper
from recording.util import mkdir, pkl_write, pkl_read
from tqdm.contrib.concurrent import process_map  # or thread_map
import multiprocessing
import cv2
import tqdm
import random


def process_sequence(seq_path, vis=False):
    with MediaPipeWrapper() as wrapper:
        seq_reader = SequenceReader(seq_path)

        save_dict = dict()
        save_path = os.path.join('data', 'pose_estimates', seq_reader.participant, seq_reader.action + '.pkl')

        if os.path.exists(save_path):
            try:
                pkl_read(save_path)
                # print('Already found', save_path)
                return
            except:
                print('Failed reading, regenerating', save_path)

        print('Doing', seq_path)

        mkdir(save_path, cut_filename=True)

        for camera_idx in seq_reader.camera_ids:
            # print('Camera', camera_idx)
            camera_dict = dict()
            save_dict[camera_idx] = camera_dict
            for timestep in range(seq_reader.num_frames):
                image = seq_reader.get_img(camera_idx, timestep, to_rgb=True)
                pose = wrapper.run_img(image, vis=vis)

                camera_dict[timestep] = pose
                # print(seq_path, camera_idx, timestep, pose)

                if vis:
                    cv2.imshow('mediapipe vis', cv2.cvtColor(pose['vis'], cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
                    del pose['vis']

    pkl_write(save_path, save_dict)
    # print('Wrote to', save_path)


if __name__ == "__main__":
    DATA_DIR = ['data/processed_weak_data/*/*',
                'data/processed_weak_data_iccv/*/*',
                'data/pressurevision/*/*/*',
                ]

    all_sequences = []
    for d in DATA_DIR:
        all_sequences.extend(glob.glob(d))

    all_sequences.sort()
    # random.shuffle(all_sequences)
    print(len(all_sequences))

    parallel = True  # Set to true for increased speed, set to false for easier debugging
    if parallel:
        # pool = multiprocessing.Pool(processes=12)  # Can crash server if too many threads
        # results = pool.map(process_sequence, all_sequences)
        process_map(process_sequence, all_sequences, max_workers=24, chunksize=1)
    else:
        for p in tqdm(all_sequences):
            process_sequence(p)
