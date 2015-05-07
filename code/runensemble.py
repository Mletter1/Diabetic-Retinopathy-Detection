#! /usr/bin/env python
"""Runs ensemble of classifiers based on database features"""

import Classify
from feature import Feature, mysql_db
import numpy as np
import csv
from itertools import izip
import logging
import argparse
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

N_FOLD = 3

logger = logging.getLogger(__name__)

def get_features(query, extra):
    """Retrieve features from database"""
    #Training data query

    rows = []
    extras = []
    for feature in query.naive().iterator():
        row = np.hstack([feature.gray_hist, feature.pca])
        """
        row = np.hstack([feature.gray_hist,
                         feature.red_hist,
                         feature.green_hist,
                         feature.blue_hist,
                         feature.hue_hist,
                         feature.saturation_hist,
                         feature.value_hist])
                         feature.pca])
        """
        rows.append(row)
        extras.append(extra(feature))

    return np.vstack(rows).astype(np.float64), np.array(extras)

def get_training():
    """Get training related features and labels"""
    query = Feature.select(Feature.gray_hist, Feature.pca,\
                           Feature.label).where(Feature.label.is_null(False))
    #query = Feature.select().where(Feature.label.is_null(False))
    get_label = lambda x: x.label
    return get_features(query, get_label)

def get_test():
    """Get test related features and names"""
    query = (Feature
             .select(Feature.name, Feature.gray_hist, Feature.pca)
             .where(Feature.label.is_null(True))
             .order_by(Feature.name))
    """
    query = (Feature
             .select()
             .where(Feature.label.is_null(True))
             .order_by(Feature.name))
    """
    get_name = lambda x: x.name
    return get_features(query, get_name)

def write_predictions(output_file, test_names, predictions):
    """Write predictions to output_file"""
    with open(output_file, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|',
                            quoting=csv.QUOTE_MINIMAL)
        for name, label in izip(test_names, predictions):
            writer.writerow([name, int(label)])

'''-------------------------------------------------------------------------'''
def setup_arguments(parser):
    """Setup arguments"""
    pass

def setup_options(parser):
    """Setup options for parser"""
    help_str = "Turn on debug"
    parser.add_argument("-d", "--debug", action="store_true", default=False,
                        help=help_str)

    help_str = "Cross validate on training instead of predicting test"
    parser.add_argument("-x", "--xval", action="store_true", default=False,
                        help=help_str)

    help_str = "Results file when predicting test labels [default:%(default)s]"
    parser.add_argument("-o", "--outfile",
                        default="results.csv", help=help_str)

def validate_arguments(args):
    """Validate arguments. Return None if valid and error string if not"""
    return None

def process_normal(output_file):
    """Process test data instead of crossval data"""
    mysql_db.connect()
    try:
        logger.debug("Getting training data")
        train_features, train_labels = get_training()
        logger.debug("Getting test data")
        test_features, test_names = get_test()
    except:
        mysql_db.close()
        raise
    mysql_db.close()
    logging.debug("scaling")
    scaler = StandardScaler(copy=False)
    scaler.fit_transform(train_features)
    scaler.transform(test_features)
    logger.debug("running classifier")
    predictions = Classify.runClassifiers(train_features, train_labels,
                                          test_features)
    logger.debug("writing predictions")
    write_predictions(output_file, test_names, predictions)

def process_xval():
    """Cross validation on training data"""
    mysql_db.connect()
    try:
        logger.debug("Getting training data")
        features, labels = get_training()
        logger.debug("Retrieved training data")
    except:
        mysql_db.close()
        raise
    mysql_db.close()
    logger.debug("Calculating k-fold indices")
    skf = StratifiedKFold(labels, n_folds=N_FOLD)

    conf_mtx = None
    logger.debug("Starting cross validation")
    for train_idx, test_idx in skf:
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        logger.debug("scaling")
        scaler = StandardScaler(copy=False)
        scaler.fit_transform(X_train)
        scaler.transform(X_test)
        logger.debug("Running classifier")
        predictions = Classify.runClassifiers(X_train, y_train, X_test)
        if conf_mtx == None:
            conf_mtx = confusion_matrix(y_test, predictions)
        else:
            conf_mtx += confusion_matrix(y_test, predictions)
        print classification_report(y_test, predictions, digits=2)

    print conf_mtx


def process(crossval, output_file):
    """Gets features, trains ensemble, makes predictions, writes results"""
    if not crossval:
        process_normal(output_file)
    else:
        process_xval()

def main():
    """Setup argparse and process request"""
    description = "Retrieves feature from database and runs predictions"
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
    process(args.xval, args.outfile)


'''-------------------------------------------------------------------------'''
if __name__ == "__main__":
    main()
