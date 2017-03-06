import getopt
import sys
from os.path import join, basename

import utils
import constants as c


def run(input_path):
    print 'Train classifier'
    utils.train_model()

    # print 'Read input'
    # imgs = utils.read_input(input_path)
    #
    # print 'Extract features'

    # print 'Sliding window search'

    # print 'Clean multi-detections and false positives'

    # return imgs_superimposed

##
# TEST
##

from glob import glob
def test():
    paths = glob(join(c.TEST_DIR, '*.jpg'))
    for path in paths:
        print path

        imgs = run(path)
        # utils.display_images(imgs)

        save_path = join(c.SAVE_DIR, 'test/' + basename(path))
        utils.save(imgs, save_path)


##
# Handle command line input
##

def print_usage():
    print 'Usage:'
    print '(-p / --path=) <path/to/image/or/video> (Required.)'
    print '(-T / --test)  (Boolean flag. Whether to run the test function instead of normal run.)'

if __name__ == "__main__":
    path = None

    try:
        opts, _ = getopt.getopt(sys.argv[1:], 'p:T', ['path=', 'test'])
    except getopt.GetoptError:
        print_usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-p', '--path'):
            path = arg
        if opt in ('-T', '--test'):
            test()
            sys.exit(2)

    # if path is None:
    #     print_usage()
    #     sys.exit(2)

    run(path)
    # imgs_processed = run(path)
    #
    # # Save images. Use same filename as input, but in save directory.
    # save_path = join(c.SAVE_DIR, basename(path))
    # utils.save(imgs_processed, save_path)