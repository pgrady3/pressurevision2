import random
import cv2
import glob
from recording.sequence_reader import SequenceReader
from tqdm import tqdm


def write_movie(seq_path):
    seq_reader = SequenceReader(seq_path)

    for i in tqdm(range(seq_reader.num_frames)):
        frame = seq_reader.get_overall_frame(i, overlay_force=True)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    file_list = glob.glob('data/data_sensel_test/*/*')
    random.shuffle(file_list)

    for f in file_list:
        write_movie(f)
