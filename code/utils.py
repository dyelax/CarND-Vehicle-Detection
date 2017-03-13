import numpy as np
import cv2
import skvideo.io
from glob import glob
from os.path import join, exists, splitext
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import matplotlib.pyplot as plt
import os
from scipy.ndimage.measurements import label
import subprocess as sp

import constants as c


##
# I/O
##

def get_dir(directory):
    """
    Returns the directory, creating it if it does not exist.

    :param directory: The path for which to get the parent directory.

    :return: The directory.
    """
    try:
        os.makedirs(directory)  # Recursively creates the current directory
    except OSError:
        pass  # Directory already exists

    return directory


def get_path(path):
    """
    Returns the path, creating the parent directory it if it does not exist.

    :param path: The path to return.

    :return: The path.
    """
    parent_dir = os.path.abspath(os.path.join(path, os.pardir))
    get_dir(parent_dir)

    return path


def read_input(paths):
    """
    Reads images from a list of input paths into a numpy array. Paths can either be .png for single
    images or .mp4 for videos.

    :param paths: The list of paths to read.

    :return: A numpy array of images - the frames from each path in paths, concatenated.
    """
    frames = []

    for path in paths:
        ext = splitext(path)[1]
        assert ext == '.png' or ext == '.mp4', 'The input file must be a .png or .mp4.'

        if ext == '.png':
            # Input is a single image.
            img = cv2.imread(path)
            # turn into a 4D array so all functions can apply to images and video.
            frames.append(np.array([img]))
        else:
            # Input is a video.
            vidcap = cv2.VideoCapture(path)

            # Load frames
            frames_list = []
            while vidcap.isOpened():
                ret, frame = vidcap.read()

                if ret:
                    frames_list.append(frame)
                else:
                    break

            vidcap.release()

            frames.append(np.array(frames_list))

    return np.concatenate(frames)


def save_output(imgs, path):
    """
    Saves imgs to file. Paths can either be .png for single images or .mp4 for videos.

    :param imgs: The frames to save. A single image for .pngs, or multiple frames for .mp4s.
    :param path: The path to which the image / video will be saved.
    """
    ext = splitext(path)[1]
    assert ext == '.png' or ext == '.mp4', 'The output file must be a .png or .mp4.'

    if ext == '.png':
        # Output is a single image.
        cv2.imwrite(path, imgs[0])
    else:
        # Output is a video.
        vid_frames = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]) * 255
        vid_frames.astype(np.uint8)
        skvideo.io.vwrite(path + '.avi', vid_frames)


def save_model(model, scaler, path=c.MODEL_SAVE_PATH):
    """
    Saves a trained model to file.

    :param model: The trained scikit learn model.
    :param scaler: The scaler used to normalize the data when training.
    :param path: The filepath to which to save the model.
    """
    save_dict = {'model': model, 'scaler':scaler}
    joblib.dump(save_dict, path)


def load_model(path=c.MODEL_SAVE_PATH):
    """
    Loads a trained model from file.

    :param path: The filepath from which to load the model.

    :return: A tuple, (model, scaler).
    """
    save_dict = joblib.load(path)
    print 'Model loaded from %s' % path

    model = save_dict['model']
    scaler = save_dict['scaler']

    return model, scaler


##
# Model training
##

def convert_color(imgs, color_space):
    """
    Converts RGB images to the given color space.

    :param imgs: The RGB images to convert.
    :param color_space: The color space to which to convert the images.
                        Options: Gray, HSV, LUV, HLS, YUV, YCrCb

    :return: The color-converted versions of imgs.
    """
    assert color_space in ['Gray', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb'], \
        "Color space must be one of 'Gray', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb'"

    imgs_converted = np.empty_like(imgs)

    # Convert every image in imgs.
    for i, img in enumerate(imgs):
        if color_space == 'Gray':
            img_converted = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif color_space == 'HSV':
            img_converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif color_space == 'LUV':
            img_converted = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        elif color_space == 'HLS':
            img_converted = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        elif color_space == 'YUV':
            img_converted = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        else: # color_space == 'YCrCb':
            img_converted = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

        imgs_converted[i] = img_converted

    return imgs_converted


def get_HOG_features(imgs, num_orientations=c.NUM_ORIENTATIONS, pix_per_cell=c.PIX_PER_CELL,
                     cells_per_block=c.CELLS_PER_BLOCK, feature_vec=True, visualize=False):
    """
    Calculates the Histogram of Oriented Gradient features for the relevant region (lower half) of
    the given images.

    :param imgs: The images for which to calculate HOG features.
    :param num_orientations: The number of gradient orientation bins for the histogram.
    :param pix_per_cell: The number of pixels in a HOG cell.
    :param cells_per_block: The number of HOG cells in a block (for block normalization).
    :param feature_vec: Whether to return as a 1D array of features (True) or keep the dimensions of
                        imgs (False).
    :param visualize: Whether to return a tuple, (features, visualization img), (True) or just the
                      features (False).

    :return: The HOG features for imgs.
    """
    features = []  # Easier to use lists than np arrays because dimensions vary based on inputs.

    if visualize:
        hog_imgs = []

        for img in imgs:
            c1_features, c1_img = hog(img[:, :, 0],
                                      orientations=num_orientations,
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cells_per_block, cells_per_block),
                                      transform_sqrt=True,
                                      visualise=visualize,
                                      feature_vector=feature_vec)
            c2_features, c2_img = hog(img[:, :, 1],
                                      orientations=num_orientations,
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cells_per_block, cells_per_block),
                                      transform_sqrt=True,
                                      visualise=visualize,
                                      feature_vector=feature_vec)
            c3_features, c3_img = hog(img[:, :, 3],
                                      orientations=num_orientations,
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cells_per_block, cells_per_block),
                                      transform_sqrt=True,
                                      visualise=visualize,
                                      feature_vector=feature_vec)

            if feature_vec:
                features.append(np.concatenate([c1_features,
                                                c2_features,
                                                c3_features]))
                hog_imgs.append(np.concatenate([c1_img,
                                                c2_img,
                                                c3_img]))
            else:
                features.append(np.array([c1_features,
                                          c2_features,
                                          c3_features]))
                hog_imgs.append(np.array([c1_img,
                                          c2_img,
                                          c3_img]))

        return np.array(features), hog_imgs
    else:
        for img in imgs:
            c1_features = hog(img[:,:,0],
                              orientations=num_orientations,
                              pixels_per_cell=(pix_per_cell, pix_per_cell),
                              cells_per_block=(cells_per_block, cells_per_block),
                              transform_sqrt=True,
                              visualise=visualize,
                              feature_vector=feature_vec)
            c2_features = hog(img[:,:,1],
                              orientations=num_orientations,
                              pixels_per_cell=(pix_per_cell, pix_per_cell),
                              cells_per_block=(cells_per_block, cells_per_block),
                              transform_sqrt=True,
                              visualise=visualize,
                              feature_vector=feature_vec)
            c3_features = hog(img[:,:,2],
                              orientations=num_orientations,
                              pixels_per_cell=(pix_per_cell, pix_per_cell),
                              cells_per_block=(cells_per_block, cells_per_block),
                              transform_sqrt=True,
                              visualise=visualize,
                              feature_vector=feature_vec)

            if feature_vec:
                features.append(np.concatenate([c1_features,
                                                c2_features,
                                                c3_features]))
            else:
                features.append(np.array([c1_features,
                                          c2_features,
                                          c3_features]))

        return np.array(features)

def get_color_bin_features(imgs, shape=c.COLOR_BIN_SHAPE):
    """
    Calculates color bin features for the given images by downsizing and taking each pixel as
    representative of the colors of the surrounding pixels in the full-size image.

    :param imgs: The images for which to calculate color bin features.
    :param shape: A tuple, (height, width) - the shape to which imgs should be downsized.

    :return: The color bin features for imgs.
    """
    # Sized to hold the ravelled pixels of each downsized image.
    features = np.empty([imgs.shape[0], shape[0] * shape[1] * imgs.shape[3]])

    # Resize and ravel every image to get color bin features.
    for i, img in enumerate(imgs):
        features[i] = cv2.resize(img, shape).ravel()

    return features

def get_color_hist_features(imgs, nbins=c.NUM_HIST_BINS, bins_range=c.HIST_BINS_RANGE):
    """
    Calculates color histogram features for each channel of the given images.

    :param imgs: The images for which to calculate a color histogram.
    :param nbins: The number of histogram bins to sort the color values into.
    :param bins_range: The range of values over all bins.

    :return: The color histogram features of each channel for every image in imgs.
    """
    num_features = imgs.shape[-1] * nbins
    hist_features = np.empty([len(imgs), num_features])

    for i, img in enumerate(imgs):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)

        # Concatenate the histograms into a single feature vector
        hist_features[i] = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    return hist_features
#
# def normalize(feature_vecs):
#     """
#     Normalizes the feature vectors.
#
#     :param feature_vecs: The feature vectors to normalize.
#
#     :return: A feature vector representing the normalized features in feature_vecs.
#     """
#     scaler = StandardScaler().fit(feature_vecs)
#     return scaler.transform(feature_vecs)


def get_feature_vectors(imgs, color_space=c.COLOR_SPACE):
    """
    Gets the feature vectors for the given images. Used to featurize training images.

    :param imgs: The images for which to get feature vectors.
    :param color_space: The color space to which to convert the images.

    :return: The feature vectors (with HOG and color features) for imgs.
    """
    imgs_cvt = convert_color(imgs, color_space)

    HOG_features = get_HOG_features(imgs_cvt)
    color_bin_features = get_color_bin_features(imgs_cvt)
    color_hist_features = get_color_hist_features(imgs_cvt)

    # Concatenate feature vectors for each image.
    features = np.concatenate([HOG_features, color_hist_features, color_bin_features], axis=1)

    return features


def resize_batch(imgs, shape):
    """
    Resizes multiple images to the given shape

    :param imgs: The images to resize.
    :param shape: The shape to which to resize the images.

    :return: The images, resized to the given shape.
    """
    imgs_resized = np.empty((len(imgs), shape[1], shape[0], 3))

    for i, img in enumerate(imgs):
        imgs_resized[i] = cv2.resize(img, shape)

    return imgs_resized


def get_sliding_window_preds(imgs, model, scaler, y_start=400, y_stop=656, cell_stride=2,
                             window_scale=1, color_space=c.COLOR_SPACE):
    """
    Gets detection predictions from a trained model on a sliding window over the given images.

    :param imgs: The images for which to get predictions.
    :param model: The model to make predictions.
    :param y_start: The pixel value on the y axis at which to start searching for cars (Top of
                    search window).
    :param y_stop: The pixel value on the y axis at which to stop searching for cars (Bottom of
                   search window).
    :param cell_stride: The stride of the sliding window, in HOG cells.
    :param window_scale: The scale of the sliding window relative to the training window size
                         (64x64).
    :param color_space: The color space to which to convert the images.

    :return: A heatmap of the predictions at each sliding window location for all images in imgs.
    """
    heatmaps = np.zeros(imgs.shape[:3])

    imgs_cvt = convert_color(imgs, color_space)
    imgs_cropped = imgs_cvt[:, y_start:y_stop, :, :]

    height, width = imgs_cropped.shape[1:3]

    # Scale the images based on the window scale. Because the model was trained on 64x64 patches
    # of HOG features, we still need that many features, so if we want a smaller window, we need
    # To size up the image so 64x64 is relatively smaller.
    if window_scale != 1:
        imgs_cropped = resize_batch(imgs_cropped, (int(width / window_scale),
                                                   int(height / window_scale)))
        height, width = imgs_cropped.shape[1:3]

    num_blocks_x = (width // c.PIX_PER_CELL) - 1
    num_blocks_y = (height // c.PIX_PER_CELL) - 1
    nblocks_per_window = int(c.HOG_WINDOW_WIDTH // c.PIX_PER_CELL) - 1

    num_steps_x = (num_blocks_x - nblocks_per_window) // cell_stride
    num_steps_y = (num_blocks_y - nblocks_per_window) // cell_stride

    # Compute hog features over whole image for efficiency.
    hog_features = get_HOG_features(imgs_cropped, feature_vec=False)

    for x_step in range(num_steps_x):
        for y_step in range(num_steps_y):
            y_pos = y_step * cell_stride
            x_pos = x_step * cell_stride

            # Extract HOG for this patch
            c1_HOG = hog_features[:,
                                  0,
                                  y_pos:y_pos + nblocks_per_window,
                                  x_pos:x_pos + nblocks_per_window]
            c2_HOG = hog_features[:,
                                  1,
                                  y_pos:y_pos + nblocks_per_window,
                                  x_pos:x_pos + nblocks_per_window]
            c3_HOG = hog_features[:,
                                  2,
                                  y_pos:y_pos + nblocks_per_window,
                                  x_pos:x_pos + nblocks_per_window]
            c1_HOG_ravelled = np.reshape(c1_HOG, [len(imgs), -1])
            c2_HOG_ravelled = np.reshape(c2_HOG, [len(imgs), -1])
            c3_HOG_ravelled = np.reshape(c3_HOG, [len(imgs), -1])

            patch_HOG = np.concatenate((c1_HOG_ravelled, c2_HOG_ravelled, c3_HOG_ravelled), axis=1)

            xleft = x_pos * c.PIX_PER_CELL
            ytop = y_pos * c.PIX_PER_CELL

            # Extract the image patch
            patch = imgs_cropped[:,
                                 ytop:ytop + c.HOG_WINDOW_WIDTH,
                                 xleft:xleft + c.HOG_WINDOW_WIDTH]

            # Get color features
            patch_color_bins = get_color_bin_features(patch)
            patch_color_hists = get_color_hist_features(patch)

            # Combine and normalize features
            patch_features = np.concatenate([patch_HOG, patch_color_bins, patch_color_hists],
                                            axis=1)
            patch_features_norm = scaler.transform(patch_features)

            # Make prediction
            patch_preds = model.predict(patch_features_norm)
            # Reshape so it can be broadcast with the 3D heatmaps array.
            patch_preds = np.reshape(patch_preds, [len(patch_preds), 1, 1])

            # Get the patch coordinates relative to the original image scale
            xleft_abs = np.int(xleft * window_scale)
            ytop_abs = np.int(ytop * window_scale) + y_start
            window_width_abs = np.int(c.HOG_WINDOW_WIDTH * window_scale)

            # Add prediction to the heatmap
            heatmaps[:, ytop_abs :ytop_abs + window_width_abs,
                        xleft_abs:xleft_abs + window_width_abs] += patch_preds

    print np.amax(heatmaps, axis=(1, 2))

    return heatmaps


def threshold_heatmaps(heatmaps, threshold=7):
    """
    Clean up the heatmaps by removing any values below the threshold.

    :param heatmaps: The heatmaps to threshold.
    :param threshold: The value below which to remove.

    :return: The thresholded heatmaps.
    """
    heatmaps[heatmaps < threshold] = 0
    return heatmaps


def rolling_threshold(heatmaps, threshold=4, hist_len=6):
    """
    Clean up the heatmaps by removing any values below the threshold over an average of the
    past hist_len frames.

    :param heatmaps: The heatmaps to threshold.
    :param threshold: The average value below which to remove.
    :param hist_len: How many frames in the past to consider.

    :return: The thresholded heatmaps.
    """
    class RingBuffer():
        "A 1D ring buffer using numpy arrays"

        def __init__(self, shape):
            self.shape = shape
            self.data = np.zeros(self.shape, dtype=np.float32)
            self.index = 0

        def extend(self, x):
            "adds array x to ring buffer"
            self.data[self.index] = x
            self.index = (self.index + 1) % self.shape[0]

    cleaned_heatmaps = np.empty_like(heatmaps)

    hist = RingBuffer((hist_len,) + heatmaps.shape[1:])
    for i, heatmap in enumerate(heatmaps):
        hist.extend(heatmap)

        mean_heatmap = np.mean(hist.data, axis=0)

        # Special case to average the first hist_len frames
        if i < hist_len:
            mean_heatmap *= hist_len
            mean_heatmap /= i + 1

        heatmap[mean_heatmap < threshold] = 0
        cleaned_heatmaps[i] = heatmap


    return cleaned_heatmaps


def segment_cars(heatmaps):
    """
    Get a map of where each car is located.

    :param heatmaps: The car detection heatmaps to use for segmentation.

    :return: A tuple (segmentation maps, num_cars).
    """
    segmentation_maps = np.empty_like(heatmaps)
    num_cars = []

    for i, heatmap in enumerate(heatmaps):
        frame_label = label(heatmap)
        segmentation_maps[i] = frame_label[0]
        num_cars.append(frame_label[1])

    return segmentation_maps, num_cars


def draw_boxes(imgs, segmentation_maps, num_cars):
    """
    Draw bounding boxes around each car in the images.

    :param imgs: The original frames.
    :param segmentation_maps: The segmentation maps of cars in each frame.
    :param num_cars: The number of cars detected in each frame.

    :return: The images, superimposed with bounding boxes.
    """
    imgs_superimposed = imgs.copy()
    for i, img in enumerate(imgs):
        overlay = img.copy()
        for car_num in xrange(1, num_cars[i] + 1):
            # Find pixels with each car_number label value
            nonzero = np.where(segmentation_maps[i] == car_num)

            # Identify x and y values of those pixels
            nonzero_y = np.array(nonzero[0])
            nonzero_x = np.array(nonzero[1])

            # Define a bounding box based on min/max x and y
            box = ((np.min(nonzero_x), np.min(nonzero_y)), (np.max(nonzero_x), np.max(nonzero_y)))

            # Draw the box on the image
            color = (228, 179, 0)
            cv2.rectangle(imgs_superimposed[i], box[0], box[1], color, 2)
            cv2.rectangle(overlay, box[0], box[1], color, -1)

        cv2.addWeighted(imgs_superimposed[i], 0.8, overlay, 0.2, 0)

    return imgs_superimposed

def get_train_test_data(train_frac=0.66):
    """
    Loads the train and test images from file, shuffles and splits them into train and test sets.

    :param train_frac: The percentage of images to use as training data.
                       (The rest will be testing data).

    :return: A tuple of tuples, ((images train, labels train), (images test, labels test)).
    """
    car_paths = glob(join(c.DATA_DIR, 'vehicles', '*', '*.png'))
    non_car_paths = glob(join(c.DATA_DIR, 'non-vehicles', '*', '*.png'))

    # paths = np.concatenate([car_paths, non_car_paths])
    # print 'Read Input'
    # imgs = read_input(paths)
    # print 'Get Features'
    # inputs = get_feature_vectors(imgs)
    #
    # # Normalize the features
    # scaler = StandardScaler().fit(inputs)
    # inputs_norm = scaler.transform(inputs)

    # labels = np.concatenate([np.ones(len(car_paths)), np.zeros(len(non_car_paths))])
    #
    # inputs_train, inputs_test, labels_train, labels_test = train_test_split(inputs_norm, labels,
    #                                                                         train_size=train_frac)

    # # split in order so there is less bleed between train and test sets
    # split_i_car = int(len(car_paths) * train_frac)
    # split_i_non = int(len(non_car_paths) * train_frac)
    #
    # car_train = inputs_norm[:split_i_car]
    # car_test = inputs_norm[split_i_car:len(car_paths)]
    # non_train = inputs_norm[len(car_paths):len(car_paths) + split_i_non]
    # non_test = inputs_norm[len(car_paths) + split_i_non:]
    #
    # inputs_train = np.concatenate([car_train, non_train])
    # labels_train = np.concatenate([np.ones([len(car_train)]), np.zeros(len(non_train))])
    # inputs_train, labels_train = zip(*np.random.permutation(zip(inputs_train, labels_train)))
    #
    # inputs_test = np.concatenate([car_test, non_test])
    # labels_test = np.concatenate([np.ones([len(car_test)]), np.zeros(len(non_test))])
    # inputs_test, labels_test = zip(*np.random.permutation(zip(inputs_test, labels_test)))


    car_paths_train = glob(join(c.DATA_DIR, 'vehicles', 'GTI*', '*.png'))
    car_paths_test = glob(join(c.DATA_DIR, 'vehicles', 'KITTI*', '*.png'))
    non_paths_train = glob(join(c.DATA_DIR, 'non-vehicles', 'GTI*', '*.png'))
    non_paths_test = glob(join(c.DATA_DIR, 'non-vehicles', 'Extras', '*.png'))

    print 'Read Input'
    car_imgs_train = read_input(car_paths_train)
    car_imgs_test = read_input(car_paths_test)
    non_imgs_train = read_input(non_paths_train)
    non_imgs_test = read_input(non_paths_test)

    imgs_train = np.concatenate([car_imgs_train, non_imgs_train])
    imgs_test = np.concatenate([car_imgs_test, non_imgs_test])

    labels_train = np.concatenate([np.ones([len(car_imgs_train)]), np.zeros([len(non_imgs_train)])])
    labels_test = np.concatenate([np.ones([len(car_imgs_test)]), np.zeros([len(non_imgs_test)])])

    print 'Get Features'
    inputs_train = get_feature_vectors(imgs_train)
    inputs_test = get_feature_vectors(imgs_test)

    inputs_train, labels_train = zip(*np.random.permutation(zip(inputs_train, labels_train)))
    inputs_test, labels_test = zip(*np.random.permutation(zip(inputs_test, labels_test)))

    # Normalize the features
    inputs = np.concatenate([inputs_train, inputs_test])
    scaler = StandardScaler().fit(inputs)

    inputs_train_norm = scaler.transform(inputs_train)
    inputs_test_norm = scaler.transform(inputs_test)

    return (inputs_train_norm, labels_train), (inputs_test_norm, labels_test), scaler


def train_model(load=True, load_path=c.MODEL_SAVE_PATH, save=True, save_path=c.MODEL_SAVE_PATH):
    """
    Returns a trained model. Trains a new model if load = False or no saved model exists. Otherwise,
    loads and returns the saved model from file.

    :param load: Whether to load a previously-trained model from load_path.
    :param load_path: The path from which to load a trained model if load = True.
    :param save: Whether to save the trained model to save_path.
    :param save_path: The path to which to save the trained model if save - True.

    :return: A model trained to classify car images vs non-car images.
    """
    # If there is a previously trained model and we want to use that, load and return it.
    if load and exists(load_path):
        print 'Loading pretrained model...'
        return load_model(path=load_path)

    # Otherwise, train a new model.
    print 'Get Data'
    (inputs_train, labels_train), (inputs_test, labels_test), scaler = get_train_test_data()

    # print 'Train Model'
    # model = LinearSVC()
    # model.fit(inputs_train, labels_train)
    #
    # # Test the model.
    # test_score = model.score(inputs_test, labels_test)
    # print 'Test Accuracy: ', test_score

    print 'Full Train'
    # Train on all the data
    inputs = np.concatenate([inputs_train, inputs_test])
    labels = np.concatenate([labels_train, labels_test])
    inputs, labels = zip(*np.random.permutation(zip(inputs, labels)))

    model = LinearSVC()
    model.fit(inputs, labels)

    print 'Save Model'
    if save:
        save_model(model, scaler, path=save_path)

    return model, scaler


##
# Testing
##

def arr2bar(arr):
    """
    Displays an array as a bar graph, where each element is the value of one bar.

    :param arr: The array to display.
    """
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(arr)), arr, 1)
    plt.show()


def display_images(imgs):
    """
    Displays an image and waits for a keystroke to dismiss and continue.

    :param imgs: The images to display
    """
    for img in imgs:
        # Conversion for masks
        if img.dtype == bool:
            img = np.uint8(img) * 255

        cv2.imshow('image', img)
        cv2.moveWindow('image', 0, 0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# def heatmaps2img(heatmaps):
#     """
#     Scales heatmaps so 0 is 0% opacity and the 10 is 100% opacity.
#
#     :param heatmaps: The heatmaps to visualize
#     """
#     scaled_heatmaps = np.zeros(heatmaps.shape + (4,))
#     scaled_heatmaps[:, :, :, 1] = 1  # Make it green
#
#     scaled_heatmaps[:, :, :, 3] = 0
#
#     return np.float32(scaled_heatmaps)

def heatmaps2img(heatmaps):
    """
    Scales heatmaps so 0 is 0% opacity and the 10 is 100% opacity.

    :param heatmaps: The heatmaps to visualize
    """
    scaled_heatmaps = 255 * heatmaps / 16.
    scaled_heatmaps = np.reshape(scaled_heatmaps, scaled_heatmaps.shape + (1,))
    scaled_heatmaps = np.concatenate([np.ones_like(scaled_heatmaps),
                                      np.zeros_like(scaled_heatmaps),
                                      scaled_heatmaps], axis=3)

    return np.uint8(scaled_heatmaps)

def heatmap_overlay(imgs, heatmaps):
    """
    Overlays heatmaps onto their original images.

    :param imgs: The original images.
    :param heatmaps: The car detection heatmaps of imgs.

    :return: Imgs with heatmaps overlain.
    """
    heatmap_imgs = heatmaps2img(heatmaps)

    imgs_superimposed = np.empty_like(imgs)
    for i in xrange(len(imgs)):
        imgs_superimposed[i] = cv2.addWeighted(heatmap_imgs[i], 1, imgs[i], 0.2, 0)

    return imgs_superimposed
