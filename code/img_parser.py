#!/usr/bin/env python
# coding=utf-8
__author__ = 'Matthew Letter'
import sys
import os
import copy
import traceback
import optparse
import time
import cPickle as pickle
import scipy
import numpy
from PIL import Image

DEBUG = False
IMG_DIR = './data'
STANDARD_SIZE = (256, 256)


def get_doc():
    doc = """
    SYNOPSIS

        main [-h,--help] [-v,--verbose] [--version]

    DESCRIPTION

        This is used to process image files into a 1d array

    EXAMPLES
        print out help message:
            python img_parser.py.py -h

    EXIT STATUS

        0 no issues
        1 unknown error
        2 improper params

    AUTHOR

        Name Matthew Letter mletter1@unm.edu

    LICENSE

        The MIT License (MIT)

        Copyright (c) 2015 Matthew Letter

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.

    VERSION

        v0.1
    """
    return doc


def img_to_matrix(filename, verbose=False):
    """
    takes a filename and turns it into a numpy array of RGB pixels
    """
    img = Image.open(filename)
    if verbose == True:
        print "changing size from %s to %s" % (str(img.size), str(STANDARD_SIZE))
    img = img.resize(STANDARD_SIZE)
    img = list(img.getdata())
    img = map(list, img)
    img = numpy.array(img)
    return img


def flatten_image(img):
    """
    takes in an (m, n) numpy array and flattens it
    into an array of shape (1, m * n)
    """
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]


def process_images():
    """
    recurses through all images and reduces image to a 1d array
    """
    images = [IMG_DIR + f for f in os.listdir(IMG_DIR)]
    labels = []

    data = []
    for image in images:
        img = img_to_matrix(image)
        img = flatten_image(img)
        data.append(img)
    data = numpy.array(data)


if __name__ == '__main__':
    """
    determine running params
    """
    global options, args
    try:
        start_time = time.time()
        parser = optparse.OptionParser(formatter=optparse.TitledHelpFormatter(), usage=get_doc(),
                                       version='%prog 0.1')
        parser.add_option('-v', '--verbose', action='store_true', default=False, help='verbose output')
        parser.add_option('-p', '--process', action='store_true', default=False, help='process image files')
        # get the options and args
        (options, args) = parser.parse_args()
        # determine what to do with the options supplied by the user
        if options.verbose:
            DEBUG = True
        print "options ", options
        print "args", args
        print "start time: " + time.asctime()
        if options.process:
            if args:  # truthy check
                IMG_DIR = args[0]
        # setup a standard image size; this will distort some images but will get everything into the same shape
        process_images()
        print "finish time: " + time.asctime()
        print 'TOTAL TIME IN MINUTES:',
        print (time.time() - start_time) / 60.0
        # smooth exit if no exceptions are thrown
        sys.exit(0)

    except KeyboardInterrupt, e:  # Ctrl-C
        raise e
    except SystemExit, e:  # sys.exit()
        raise e
    except Exception, e:  # unknown exception
        print 'ERROR, UNEXPECTED EXCEPTION'
        print str(e)
        traceback.print_exc()
        os._exit(1)