__author__ = 'jiananyue'

import csv
import numpy as np
import pylab

def getDataMatrix(file):
    dataOrigin = np.genfromtxt(file, delimiter=',')
    dataOrigin = dataOrigin[1:, 2:]
    target = dataOrigin[:, -1]
    (n,m) = np.shape(dataOrigin)
    dataMatrix = np.ones((n, m + 1))
    #add a column of ones for constant w0
    dataMatrix[:, :-2] = dataOrigin[:, :-1]
    #Put target back to the last column of matrix
    dataMatrix[:, -1] = target
    return dataMatrix

def getDataMatrixPart2(file):
    input = csv.reader(file, delimiter=',')
    dataOrigin = np.array(list(input))
    dataOrigin = dataOrigin[1:, :]
    target = dataOrigin[:, 0]
    dataOrigin = dataOrigin[:, 2:]
    dataOrigin = np.c_[dataOrigin, target]
    dataOrigin = np.array(dataOrigin, dtype=float)
    return dataOrigin


#split dataset at ratio 9:1
def splitDataSet(data):
    np.random.shuffle(data)
    (n, m) = data.shape
    splitIndex = int((n/10)*9)
    training, test = data[:splitIndex, :], data[splitIndex:, :]
    return training, test

#z-score normalization
def Normalization(data):
    inputData = data[:, :-2]
    for col in inputData.T:
        (n,) = col.shape
        mean = np.sum(col)/n
        var = (np.sum((col - np.ones(n)*mean)**2)/(n - 1))**0.5
        #print mean, var
        col[...] = (col - np.ones(n)*mean)/var
    NormalData = np.append(inputData, data[:,-2:],axis=1)
    NormalizedTrain, NormalizedTest = splitDataSet(NormalData)
    return NormalizedTrain, NormalizedTest

def min_max_Normalization(data):
    inputData = data[:, :-2]
    for col in inputData.T:
        (n,) = col.shape
        max = np.max(col)
        min = np.min(col)
        col[...] = (col - np.ones(n)*min)/(max - min)
    NormalData = np.append(inputData, data[:,-2:],axis=1)
    return NormalData

#Least-squares linear regression without regularization
def leastSquareMethod(dataSet):
    X = dataSet[:, :-1]
    Y = dataSet[:, -1]
    a = np.dot(np.transpose(X),X)
    b = np.dot(np.transpose(X),Y)
    model = np.dot(np.array(np.mat(a).I),b)
    return model

#Ridge regression with penalty size p
def ridgeRegression(dataSet, p):
    X = dataSet[:, :-1]
    Y = dataSet[:, -1]
    a = np.dot(np.transpose(X),X)
    b = np.dot(np.transpose(X),Y)
    (m, m) = a.shape
    I = np.eye(m) * p
    #print "p is" , p
    model = np.dot(np.array(np.mat(a + I).I),b)
    return model


def errorEvaluation(model, testSet):
    testData = testSet[:, :-1]
    testTarget = testSet[:, -1]
    a = testTarget - np.dot(testData, model)
    error = np.dot(np.transpose(a), a)
    (n,)= testTarget.shape
    error = error / n
    return error

#K-fold cross validation
def crossValidation(k, data, p):
        (n, m) = data.shape
        #n1 is the number of instance in each of the first to (k-1)th fold
        #The rest instances is in the kth fold, that is, n2 instances.
        n1 = n / k
        n2 = n - n1 * (k - 1)
        #err array keep the validation result of each turn
        err = np.zeros(k)
        #set aside the first subset:
        validation = data[:n1, :]
        training = data[n1:, :]
        model = ridgeRegression(training, p)
        err[0] = errorEvaluation(model, validation)
        #Take turns to set aside one of the subset between the first and the last subset
        for i in range(1, k-1):
            validation = data[i*n1:(i+1)*n1,:]
            training = np.append(data[0:i*n1, :], data[(i+1)*n1:, :], axis = 0)
            model = ridgeRegression(training, p)
            err[i] = errorEvaluation(model, validation)
        #set aside the last subset:
        validation = data[-n2:, :]
        training = data[:-n2, :]
        model = ridgeRegression(training, p)
        err[k-1] = errorEvaluation(model, validation)
        averageErr = np.sum(err)/k
        return averageErr

#This function use 5-fold cross validation on different size of penalty of ridge regression
#It plot the training error according to different parameter
def crossValidationOnPenaltySize(trainSet):
    penalties = np.array([0.1,1,10,100,1000,10000,100000,1000000])
    averageValidationErr = np.zeros(8)
    for k in range(8):
        #5-fold cross validation on penalty size p
        averageValidationErr[k] = crossValidation(5, trainSet, penalties[k])


    #pylab.semilogx(penalties,averageValidationErr,'ro',label='validationErr')
    #pylab.xlabel("Penalty Size")
    #pylab.xlim(10**(-2),10**7)
    #pylab.legend(loc='upper right')
    #pylab.show()
    return penalties[np.argmin(averageValidationErr)]


#This function plot the average validation error calculated by cross validation using different fold number k.
#The penalty size is 100
def differentFoldSize(NormalizedTrain):
    averageValidationErr = np.zeros(10)
    for k in range(10):
        averageValidationErr[k] = crossValidation(k+1, NormalizedTrain,1000)
    xAxis = np.arange(1,11,1)
    pylab.plot(xAxis,averageValidationErr,'bo',label='validationErr')
    pylab.plot(xAxis,averageValidationErr,'b')
    pylab.xlabel("Number of fold in cross validation")
    pylab.legend(loc='upper right')
    pylab.xlim(0,12)
    pylab.show()

#This function implements the part1 of the mini-project
def Part1():
    print "----------------------Part1-----------------------"
    dataFile = "OnlineNewsPopularity.csv"
    dataMatrix = getDataMatrix(dataFile)
    trainSet, testSet = splitDataSet(dataMatrix)
    model = leastSquareMethod(trainSet)
    cfTrainErr = '%.3e'%errorEvaluation(model, trainSet)
    cfTestErr = '%.3e'%errorEvaluation(model, testSet)
    print cfTrainErr, "\tClosed-form trainErr"
    print cfTestErr, "\tClosed-form testErr"
    NormalizedTrain, NormalizedTest = Normalization(dataMatrix)
    penalty = crossValidationOnPenaltySize(NormalizedTrain)
    print "Best penalty size is: ", penalty
    #differentFoldSize(NormalizedTrain)
    #After experiment on penalty size and fold size,
    #We decide to train model by: penalty size=1000, k=5
    approvedModel = ridgeRegression(NormalizedTrain, penalty)
    rrTrainErr = '%.3e'%errorEvaluation(approvedModel, NormalizedTrain)
    rrTestErr = '%.3e'%errorEvaluation(approvedModel, NormalizedTest)
    print rrTrainErr, "\tRidge-regression trainErr"
    print rrTestErr, "\tRidge-regression trainErr"
    print "--------------------------------------------------"
    return cfTrainErr, cfTestErr, rrTrainErr, rrTestErr

def Part2():
    print "----------------------Part2-----------------------"
    dataFile_Part2 = open("final.csv",'r')
    dataMatrixPart2 = getDataMatrixPart2(dataFile_Part2)
    trainSet2, testSet2 = splitDataSet(dataMatrixPart2)
    model2 = leastSquareMethod(trainSet2)
    cfTrainErr = '%.3e'%errorEvaluation(model2, trainSet2)
    cfTestErr = '%.3e'%errorEvaluation(model2, testSet2)
    print cfTrainErr, "\tClosed-form trainErr"
    print cfTestErr, "\tClosed-form testErr"
    NormalizedTrain, NormalizedTest = Normalization(dataMatrixPart2)
    penalty = crossValidationOnPenaltySize(NormalizedTrain)
    print "Best penalty size is: ", penalty
    approvedModel = ridgeRegression(NormalizedTrain, penalty)
    rrTrainErr = '%.3e'%errorEvaluation(approvedModel, NormalizedTrain)
    rrTestErr = '%.3e'%errorEvaluation(approvedModel, NormalizedTest)
    print rrTrainErr, "\tRidge-regression trainErr"
    print rrTestErr, "\tRidge-regression trainErr"
    print "--------------------------------------------------"
    return cfTrainErr, cfTestErr, rrTrainErr, rrTestErr

def main():
    #----------------------Part1-----------------------
    #for 10 runs of project part1
    #plot the train and test error of both closed-form and ridge-regression solution
    cfTrainErr = np.zeros(10)
    cfTestErr = np.zeros(10)
    rrTrainErr = np.zeros(10)
    rrTestErr = np.zeros(10)
    for i in range(10):
        cfTrainErr[i], cfTestErr[i], rrTrainErr[i], rrTestErr[i] = Part1()
    pylab.semilogy(np.arange(10),cfTrainErr,'bo',label='cfTrainErr')
    pylab.semilogy(np.arange(10),cfTestErr,'ro', label='cfTestErr')
    pylab.semilogy(np.arange(10),rrTrainErr,'b^',label='rrTrainErr')
    pylab.semilogy(np.arange(10),rrTestErr,'r^', label='rrTestErr')
    pylab.semilogy(np.arange(10),cfTrainErr,'b')
    pylab.semilogy(np.arange(10),cfTestErr,'r')
    pylab.semilogy(np.arange(10),rrTrainErr,'b')
    pylab.semilogy(np.arange(10),rrTestErr,'r')
    pylab.xlabel("Run")
    pylab.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    pylab.show()

    #----------------------Part2-----------------------
    #for 10 runs of project part2
    #plot the train and test error of both closed-form and ridge-regression solution
    cfTrainErr = np.zeros(10)
    cfTestErr = np.zeros(10)
    rrTrainErr = np.zeros(10)
    rrTestErr = np.zeros(10)
    for i in range(10):
        cfTrainErr[i], cfTestErr[i], rrTrainErr[i], rrTestErr[i] = Part2()
    pylab.semilogy(np.arange(10),cfTrainErr,'bo',label='cfTrainErr')
    pylab.semilogy(np.arange(10),cfTestErr,'ro', label='cfTestErr')
    pylab.semilogy(np.arange(10),rrTrainErr,'b^',label='rrTrainErr')
    pylab.semilogy(np.arange(10),rrTestErr,'r^', label='rrTestErr')
    pylab.semilogy(np.arange(10),cfTrainErr,'b')
    pylab.semilogy(np.arange(10),cfTestErr,'r')
    pylab.semilogy(np.arange(10),rrTrainErr,'b')
    pylab.semilogy(np.arange(10),rrTestErr,'r')
    pylab.xlabel("Run")
    pylab.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    pylab.show()


if __name__ == '__main__':
    main()