#! /usr/bin/env python
"""Extract features"""
import numpy as np
from skimage import io, color
import os
import os.path
import re
import csv
import argparse
import logging
from feature import Feature, mysql_db


NUM_HIST_BINS = 256

logger = logging.getLogger(__name__)

def component_histograms(image, transform=None):
    """ Computes histograms of each component of each pixel
    Parameters:
    - image : A 3-dimensional array
    Returns:
    List of histograms with 256 bins for each component of last dimension
    """
    if transform == None:
        transform = lambda x: x

    shape = image.shape
    assert len(shape) == 3
    _, _, num_comp = shape
    return [np.histogram(transform(image[:, :, [idx]]).flatten(),
                         bins=NUM_HIST_BINS, range=(0, 256))[0]
            for idx in xrange(0, num_comp)]

def get_labels(label_file):
    """Gets the image labels from the input label_file"""
    labels = None
    with open(label_file, 'r') as infile:
        reader = csv.reader(infile)
        labels = dict((rows[0], rows[1]) for rows in reader)
    return labels


def extract(image):
    """Extract Features from input image"""
    no_black = lambda x: x[x > 0]
    r, g, b = component_histograms(image, transform=no_black)
    hsv_image = color.rgb2hsv(image)
    h, s, v = component_histograms(hsv_image, transform=no_black)
    gray_image = color.rgb2gray(image)
    gray_hist, _ = np.histogram(no_black(gray_image).flatten(),
                                NUM_HIST_BINS, range=(0, 1))
    return {'gray_hist': gray_hist, 'red_hist': r, 'green_hist': g,
            'blue_hist':b, 'hue_hist': h, 'saturation_hist': s,
            'value_hist': v}

def extract_from_dir(directory):
    """Extract features from every jpeg file in directory (recursive)"""
    image_regex = re.compile(r'.+\.jpeg$')
    for root, _, files in os.walk(directory):
        for name in files:
            if image_regex.match(name) != None:
                filename = os.path.join(root, name)
                image = io.imread(filename)
                no_ext, _ = os.path.splitext(name)
                logger.debug("extracting {0}".format(no_ext))
                features = extract(image)
                yield (no_ext, features)

def features_to_db(training_dir, test_dir, label_file):
    """Loads the training, test, and label files.  Extracts features and
    saves to db"""
    logger.debug("Getting Labels")
    labels = get_labels(label_file)
    logger.debug("Extracting training features")
    train_features = extract_from_dir(training_dir)
    logger.debug("Saving training features in db")
    for name, features in train_features:
        db_entry = Feature(name=name, label=labels[name], **features)
        db_entry.save()

    logger.debug("Extracting test features")
    test_features = extract_from_dir(test_dir)
    logger.debug("Saving test features in db")
    for name, features in test_features:
        db_entry = Feature(name=name, **features)
        db_entry.save()

"""--------------------------------------------------------------------------"""
""" Commandline setup"""
"""--------------------------------------------------------------------------"""
def setup_arguments(parser):
    """Setup arguments"""
    help_str = "Path to training directory containing .jpeg files"
    parser.add_argument("training_dir", help=help_str)

    help_str = "Path to test directory containing .jpeg files"
    parser.add_argument("test_dir", help=help_str)

    help_str = "Path to label csv file that has training labels"
    parser.add_argument("label_file", help=help_str)

def setup_options(parser):
    """Setup options for parser"""
    help_str = "Turn on debug"
    parser.add_argument("-d", "--debug", action="store_true", default=False,
                        help=help_str)

def validate_arguments(args):
    """Validate arguments. Return None if valid and error string if not"""
    if not os.path.exists(args.training_dir):
        return "{0} is not a valid directory".format(args.training_dir)

    if not os.path.exists(args.test_dir):
        return "{0} is not a valid directory".format(args.test_dir)

    if not os.path.isfile(args.label_file):
        return "{0} is not a valid file".format(args.label_file)
    return None

def main():
    """Main"""
    description = "Extract features and save to Nialls's db"
    parser = argparse.ArgumentParser(description=description)
    setup_arguments(parser)
    setup_options(parser)
    args = parser.parse_args()
    error = validate_arguments(args)
    if error != None:
        parser.error(error)
    ch = logging.StreamHandler()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARN)
        ch.setLevel(logging.WARN)
    logger.addHandler(ch)
    mysql_db.connect()
    try:
        features_to_db(args.training_dir, args.test_dir, args.label_file)
    except:
        mysql_db.close()
        raise
    mysql_db.close()

if __name__ == "__main__":
    main()
