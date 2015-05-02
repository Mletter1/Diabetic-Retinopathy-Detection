########################################################

import numpy
from scipy.stats import mode

def most_common(npArr):
    '''
    Returns the most common element in a 1D numpy array.  Used
    by the majorityVotePredictions function.
    '''
    return mode(npArr, axis=0)[0][0]


def majorityVotePredictions(classifiers, testing):
    '''
    Returns a list of predictions by applying each classifier
    to the testing set and taking a majority vote of the ensemble.
    '''
    predictions = numpy.zeros( shape=(len(testing), len(classifiers)) )
    
    for c in range(0, len( classifiers ) ):
        predictions[:, c] = classifiers[c].predict( testing )
    
    most_common_predictions = numpy.apply_along_axis( most_common, axis=0, arr = predictions )
    
    return most_common_predictions
    


def runClassifiers(training, train_labels, testing):
    '''
    Trains ensemble of classifiers on the given training set with corresponding
    train_labels.  Then, tests that ensemble on the given testing set and test_labels.
    Returns a list of predicted labels.
    '''
    
    classifiers = buildClassifiers(training, train_labels)
    
    return majorityVotePredictions(classifiers, testing)



def buildClassifiers(training, train_labels):
    '''
    Builds an ensemble (list) of classifiers from the given training data.
    The classifiers are :   < FILL ME IN >
    '''
    
    pass

