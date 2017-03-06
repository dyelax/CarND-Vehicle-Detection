from os import makedirs
from os.path import join, exists
import numpy as np


def get_dir(directory):
    """
    Creates the given directory if it does not exist.

    @param directory: The path to the directory.
    @return: The path to the directory.
    """
    if not exists(directory):
        makedirs(directory)
    return directory

DATA_DIR = '../data/'
CALIBRATION_DIR = join(DATA_DIR, 'camera_cal/')
CALIBRATION_DATA_PATH = join(CALIBRATION_DIR, 'calibration_data.p')
TEST_DIR = join(DATA_DIR, 'test_images/')

SAVE_DIR = '../output_images/'
MODEL_SAVE_PATH = join(SAVE_DIR, 'model.pkl')

IMG_WIDTH_TRAIN = HOG_WINDOW_WIDTH = 64.

IMG_HEIGHT = 720.
IMG_WIDTH = 1280.

# The number of most recent elements from the fit history to consider when looking for new lines.
RELEVANT_HIST = 5