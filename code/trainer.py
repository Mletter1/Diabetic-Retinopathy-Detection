#!/usr/bin/env python
# coding=utf-8
__author__ = 'matthewletter'
import sys
import os
import copy
import traceback
import optparse
import time
import cPickle as pickle
from sklearn import tree
import scipy
import numpy
import csv
import glob
from PIL import Image


DEBUG = False


def get_doc():
    doc = """
    SYNOPSIS

        main [-h,--help] [-v,--verbose] [--version]

    DESCRIPTION

        This is used to process image files into a 1d array

    EXAMPLES
        print out help message:
            python trainer.py -h

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


def load_pickle(file_path="train.p"):
    """
    load pickle data.p file
    ................................................

    :parameter: directory_name: name of the directory you wish to start from

    :returns:
        processed_train_images: map of {name:  numpy image array}
        labels: map of {name: class}
    """
    if DEBUG:
        print "pickle load"
    # load pickle data
    data_load = pickle.load(open(file_path, "rb"))
    if DEBUG:
        for element in data_load:
            print
            print "length of each pickle load: ", len(element)
    # get data values
    return data_load[0], data_load[1]


def get_classifiers(sample=list(), class_list=list()):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(sample, class_list)
    return clf


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
        parser.add_option('-l', '--load', action='store_true', default=False, help='process image files')
        # get the options and args
        (options, args) = parser.parse_args()
        # determine what to do with the options supplied by the user
        if options.verbose:
            DEBUG = True

        print "options ", options, "\nargs", args, "\nstart time: " + time.asctime()

        if options.load:
            if args:  # truthy check
                IMG_DIR = args[0]
            processed_train_images = {}
            processed_test_images = {}
            labels = {}
            processed_train_images, labels = load_pickle("train.p")
            processed_test_images, labels = load_pickle("test.p")
            sample = list()
            class_list = list()
            result = list()
            for key in processed_train_images:
                sample.append(processed_train_images[key])
                class_list.append(labels[key])
            clf = get_classifiers(sample, class_list)

            with open('results.csv', 'wb') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for key in processed_test_images:
                    result = clf.predict([processed_test_images[key]])[0]
                    spamwriter.writerow([key, result])

            print result
        print "finish time: " + time.asctime(), '\nTOTAL TIME IN MINUTES:', (time.time() - start_time) / 60.0
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