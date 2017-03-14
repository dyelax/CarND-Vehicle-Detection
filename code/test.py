import numpy as np
import cv2
import skvideo.io
from glob import glob
from os.path import join, exists, splitext
import pickle
import matplotlib.pyplot as plt

import constants as c
from utils import get_HOG_features, display_images, convert_color


car = cv2.imread('../examples/car.png')
non = cv2.imread('../examples/non-car.png')

car, non = convert_color([car, non], 'HLS')

HOGs = get_HOG_features([car, non], visualize=True, feature_vec=False)

car_HOG_img = HOGs[1][0][0]
non_HOG_img = HOGs[1][1][0]

# display_images([car_HOG_img, non_HOG_img])

cv2.imwrite('../examples/car_HOG.png', np.uint8(car_HOG_img) * 255)
cv2.imwrite('../examples/non_HOG.png', np.uint8(non_HOG_img) * 255)
