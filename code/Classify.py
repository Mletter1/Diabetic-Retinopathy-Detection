########################################################

import numpy
from scipy.stats import mode
from sklearn import tree
from sklearn import svm
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

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
    
    most_common_predictions = numpy.apply_along_axis( most_common, axis=1, arr = predictions )
    
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
    The classifiers are :   decision tree, 
    '''
    
    # decision tree, SVM, log. regression, fnn (feed-forward neural net), 
    # each as separate function returning the classifier 
    knn = makeKNN(training, train_labels)
    print "Classify: finished building KNN"
    nb = makeMultinomialNB(training, train_labels)
    print "Classify: finished building naive bayes"
    svmc = makeSVM(training, train_labels)
    print "Classify: finished building svm"
    dt = decTree(training, train_labels)
    print "Classify: finished building decision tree"
    lr = logReg(training, train_labels)
    print "Classify: finished building logistic regression"
    
    ensemble =  [ knn, nb, svmc, dt, lr ]
    
    print "Classify: finished building classifier ensemble"
    
    return ensemble
    

def trainClassifier(clf, training, train_labels):
    clf.fit(training, train_labels)
    return clf


def decTree(training, train_labels):
    clf = tree.DecisionTreeClassifier()
    return trainClassifier(clf, training, train_labels)


def logReg(training, train_labels):
    clf = linear_model.LogisticRegression()
    return trainClassifier(clf, training, train_labels)


def makeKNN(training, train_labels):
    knn = KNeighborsClassifier()  # defaults to k = 5
    return trainClassifier(knn, training, train_labels)


def makeSVM(training, train_labels):
    clf = svm.SVC()
    return trainClassifier(clf, training, train_labels)


def makeMultinomialNB(training, train_labels):
    clf = MultinomialNB()
    return trainClassifier(clf, training, train_labels)


