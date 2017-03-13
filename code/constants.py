from os import makedirs
from os.path import join


def get_dir(directory):
    """
    Returns the directory, creating it if it does not exist.

    :param directory: The path for which to get the parent directory.

    :return: The directory.
    """
    try:
        makedirs(directory)  # Recursively creates the current directory
    except OSError:
        pass  # Directory already exists

    return directory


DATA_DIR = '../data/'
CALIBRATION_DIR = join(DATA_DIR, 'camera_cal/')
CALIBRATION_DATA_PATH = join(CALIBRATION_DIR, 'calibration_data.p')
TEST_DIR = join(DATA_DIR, 'test_images/')

SAVE_DIR = get_dir('../output_images/')
MODEL_SAVE_DIR = get_dir(join(SAVE_DIR, 'models'))
MODEL_SAVE_PATH = join(MODEL_SAVE_DIR, 'model-HLS-12-FULL.pkl')

IMG_WIDTH_TRAIN = HOG_WINDOW_WIDTH = 64

IMG_HEIGHT = 720.
IMG_WIDTH = 1280.

##
# Feature Parameters
##

COLOR_SPACE = 'HLS'

# HOG

NUM_ORIENTATIONS = 12
PIX_PER_CELL = 8
CELLS_PER_BLOCK = 2

# Color Bin

COLOR_BIN_SHAPE = (16, 16)

# Color Hist

NUM_HIST_BINS = 32
HIST_BINS_RANGE = (0, 256)