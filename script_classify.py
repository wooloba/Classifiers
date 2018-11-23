from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np

import dataloader as dtl
import classalgorithms as algs


def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct/float(len(ytest))) * 100.0

def geterror(ytest, predictions):
    return (100.0-getaccuracy(ytest, predictions))

## k-fold cross-validation
# K - number of folds
# X - data to partition
# Y - targets to partition
# classalgs - a dictionary mapping algorithm names to algorithm instances
#
# example:
classalgs = {
  'Logistic Regression': algs.LogitReg(),
    'Neural Network': algs.NeuralNet({'epochs': 100})
}

parameters = (
        #{'regwgt': 0.0, 'nh': 4},
        {'regwgt': 0.01, 'nh': 8},
        {'regwgt': 0.05, 'nh': 16},
        {'regwgt': 0.1, 'nh': 32},
                      )


def cross_validate(K, X, Y, classalgs):
    errors = {}
    numparams = len(parameters)

    for learnername in classalgs:
        errors[learnername] = np.zeros((numparams))

    set_len = len(X) / K

    for k in range(K):
        for p in range(numparams):
            params = parameters[p]

            for learnername,learner in classalgs.items():
                start = k*set_len
                Xtest_set = X[start:start+set_len]
                Ytest_set = Y[start:start+set_len]
                Xtrain_set = np.concatenate((X[0:start],X[start+set_len:len(X)]),axis=0)
                Ytrain_set =  np.concatenate((Y[0:start],Y[start+set_len:len(Y)]),axis=0)

                # Reset learner for new parameters
                learner.reset(params)
                print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                # Train model
                learner.learn(Xtrain_set, Ytrain_set)
                # Test model
                predictions = learner.predict(Xtest_set)
                error = geterror(Ytest_set, predictions)
                print ('Error for ' + learnername + ': ' + str(error))
                errors[learnername][p] = error

    for learnername, learner in classalgs.items():
        besterror = np.mean(errors[learnername][0,:])
        bestparams = 0
        for p in range(numparams):
            aveerror = np.mean(errors[learnername][p,:])
            if aveerror < besterror:
                besterror = aveerror
                bestparams = p


    best_algorithm = classalgs[p]
    return best_algorithm


if __name__ == '__main__':

    trainsize = 5000
    testsize = 5000
    numruns = 10

    classalgs = {
                'Random': algs.Classifier(),
                 'Naive Bayes': algs.NaiveBayes({'usecolumnones': False}),
                 'Naive Bayes Ones': algs.NaiveBayes({'usecolumnones': True}),
                 'Linear Regression': algs.LinearRegressionClass(),
                   'Logistic Regression': algs.LogitReg(),
                  'Neural Network': algs.NeuralNet({'epochs': 100}),
                   'LinearKernelLogitReg':algs.KernelLogitReg({'kernel':'linear','regwgt': 0.01, 'regularizer': 'None'}),
                    'HammingKernelLogitReg': algs.KernelLogitReg({'kernel': 'hamming', 'regwgt': 0.01, 'regularizer': 'None'})
    }
    numalgs = len(classalgs)

    parameters = (
        #{'regwgt': 0.0, 'nh': 4},
        {'regwgt': 0.01, 'nh': 8},
        {'regwgt': 0.05, 'nh': 16},
        {'regwgt': 0.1, 'nh': 32},
                      )
    numparams = len(parameters)

    errors = {}
    for learnername in classalgs:
        errors[learnername] = np.zeros((numparams,numruns))

    for r in range(numruns):
        '''
        To run hamming distance kernel, please comment out line 75 and uncomment line 77
        '''
        trainset, testset = dtl.load_susy(trainsize,testsize)
        #trainset, testset = dtl.load_susy_complete(trainsize,testsize)
        #trainset, testset = dtl.load_census(trainsize,testsize)

        print(('Running on train={0} and test={1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0],r))

        for p in range(numparams):
            params = parameters[p]
            for learnername, learner in classalgs.items():
                # Reset learner for new parameters
                learner.reset(params)
                print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                # Train model
                learner.learn(trainset[0], trainset[1])
                # Test model
                predictions = learner.predict(testset[0])
                error = geterror(testset[1], predictions)
                print ('Error for ' + learnername + ': ' + str(error))
                errors[learnername][p,r] = error

    for learnername, learner in classalgs.items():
        besterror = np.mean(errors[learnername][0,:])
        bestparams = 0
        for p in range(numparams):
            aveerror = np.mean(errors[learnername][p,:])
            if aveerror < besterror:
                besterror = aveerror
                bestparams = p

        # Extract best parameters
        learner.reset(parameters[bestparams])
        print ('Best parameters for ' + learnername + ': ' + str(learner.getparams()))
        print ('Average error for ' + learnername + ': ' + str(besterror) + ' +- ' + str(np.std(errors[learnername][bestparams,:])/math.sqrt(numruns)))
