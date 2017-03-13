import getopt
import sys
from os.path import join, basename

import utils
import constants as c


def run(input_paths):
    print '-' * 30
    print 'Train classifier:'
    model, scaler = utils.train_model()
    print '-' * 30

    print 'Read input'
    imgs = utils.read_input(input_paths)

    print 'Sliding window search'
    # heatmaps = utils.get_sliding_window_preds(imgs, model, scaler, window_scale=2)
    heatmaps = utils.get_sliding_window_preds(imgs, model, scaler)
    heatmaps += utils.get_sliding_window_preds(imgs, model, scaler, window_scale=1.5)
    heatmaps += utils.get_sliding_window_preds(imgs, model, scaler, window_scale=0.75,
                                               y_start=400, y_stop=500)

    print 'Clean multi-detections and false positives'
    heatmaps_clean = utils.rolling_threshold(heatmaps)
    # heatmap_overlays = utils.heatmap_overlay(imgs, heatmaps_clean)
    # utils.display_images(heatmap_overlays)
    # return heatmap_overlays
    car_segmentation, num_cars = utils.segment_cars(heatmaps_clean)

    print 'Find and draw bounding boxes'
    imgs_superimposed = utils.draw_boxes(imgs, car_segmentation, num_cars)
    # utils.display_images(imgs_superimposed)

    return imgs_superimposed

##
# TEST
##

from glob import glob
def test():
    paths = glob(join(c.TEST_DIR, '*.png'))
    imgs = run(paths)

    for i, path in enumerate(paths):
        save_path = utils.get_path(join(c.SAVE_DIR, 'test/' + basename(path)))
        utils.save_output([imgs[i]], save_path)


##
# Handle command line input
##

def print_usage():
    print 'Usage:'
    print '(-p / --path=) <path/to/image/or/video> (Required.)'
    print '(-T / --test)  (Boolean flag. Whether to run the test function instead of normal run.)'

if __name__ == "__main__":
    paths = None

    try:
        opts, _ = getopt.getopt(sys.argv[1:], 'p:T', ['paths=', 'test'])
    except getopt.GetoptError:
        print_usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-p', '--paths'):
            paths = [arg]
        if opt in ('-T', '--test'):
            test()
            sys.exit(2)

    if paths is None:
        print_usage()
        sys.exit(2)

    # Will only work for videos rn
    imgs_processed = run(paths)
    save_path = utils.get_path(join(c.SAVE_DIR, basename(paths[0])))
    utils.save_output(imgs_processed, save_path)