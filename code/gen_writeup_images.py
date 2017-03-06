import utils
import constants as c
from os.path import join

# Undistorted calibration image
calibration_img = utils.read_input('../data/camera_cal/calibration1.jpg')
camera_mat, dist_coeffs = utils.calibrate_camera()
cal_undistorted = utils.undistort_imgs(calibration_img, camera_mat, dist_coeffs)
cal_save_path = join(c.SAVE_DIR, 'undistort_cal.jpg')
utils.save(cal_undistorted, cal_save_path)

# Undistorted road image
road_img = utils.read_input('../data/test_images/test3.jpg')
road_undistorted = utils.undistort_imgs(road_img, camera_mat, dist_coeffs)
road_save_path = join(c.SAVE_DIR, 'undistort_road.jpg')
utils.save(road_undistorted, road_save_path)

# Mask image
mask = utils.get_masks(road_undistorted)
mask_save_path = join(c.SAVE_DIR, 'mask.jpg')
utils.save(mask * 255, mask_save_path)

# Birdseye transform
birdseye = utils.birdseye(mask)
birdseye_save_path = join(c.SAVE_DIR, 'birdseye.jpg')
utils.save(birdseye * 255, birdseye_save_path)

# Fit lines
find_fit = utils.visualize_find_fit(birdseye[0])
find_fit_save_path = join(c.SAVE_DIR, 'find_fit.jpg')
utils.save(find_fit, find_fit_save_path)

# Output
lines, history = utils.find_lines(birdseye)
output = utils.draw_lane(road_undistorted, lines, history)
output_save_path = join(c.SAVE_DIR, 'test3.jpg')
utils.save(output, output_save_path)