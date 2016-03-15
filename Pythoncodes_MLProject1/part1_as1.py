# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 14:16:46 2015

@author: Timardeep
"""
import pandas as pd
import csv
import numpy as np
from sklearn.cross_validation import KFold
from sklearn import preprocessing
import math,sys
from collections import defaultdict
from operator import itemgetter
from matplotlib import pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D
import random
#Linear regression with gradient descent
def preProcessWithPanda(file):
    data = pd.read_csv(file, header = 0, delimiter = ",", quoting = 3)
    return data
def read(file):
    with open(file,'rb') as csvFile:
        data = csv.reader(csvFile, delimiter = ',')
        data = list(data)
        data = np.asanyarray(data)
        data = data[1:,2:]
    return data
def splitDataSet(data):
    np.random.shuffle(data)
    (n, m) = data.shape
    splitIndex = int((n/10)*6)
    training, test = data[:splitIndex, :], data[splitIndex:, :]
    return training, test
def nomalize(data):
    data = np.asarray(data).astype(np.float)
    data_mean = np.mean(data[0:,0:-1],axis = 0)
    data_std = np.std(data[0:,0:-1], axis = 0)
    normalized_data_x = (data[0:,0:-1] - data_mean)/data_std
    n,m = data.shape
    dataa = np.ones((n,m+1))
    dataa[0:,0:-2] = normalized_data_x
    dataa[0:, -1] = data[0:, -1]
    return dataa
def noNormalize(data):
    n,m = data.shape
    data = np.asarray(data).astype(np.float)
    dataa = np.ones((n,m+1))
    dataa[0:,0:-2] = data[0:,0:-1]
    dataa[0:, -1] = data[0:, -1]
    return dataa
def crossvalidationRidgeGradient(training,k):
    # print normalized_data
    n,m = training.shape
    #print n
    numPerFold = n/k
    print numPerFold,'numPerFold'
    w0 = np.array([0.00]*(m-1))
    w0 = np.asmatrix(w0).T
    train_errs = defaultdict(lambda:defaultdict(list))
    valid_errs = defaultdict(lambda:defaultdict(list))
    #w = defaultdict(list)
    train_err_mean = defaultdict(lambda:defaultdict(list))
    valid_err_mean = defaultdict(lambda:defaultdict(list))
    lambs = [1,10,100]
    # lambs = [1, 10]
    # alphas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1 ,10, 100]
    alphas = [0.1,1,10]
    for lamb in lambs:
        print lamb,'lamb'
        for alpha in alphas:
            print alpha,'alpha'
            for i in range(k):
                print i
                # print i * numPerFold,'i*numPerFold'
                validation = training[i * numPerFold: (i+1) * numPerFold, 0: ]
                # print validation.shape,'test.shape'
                rr =range(i * numPerFold , (i+1) * numPerFold + 1)
                train = np.delete(training, rr, axis=0)
                train = np.asmatrix(train)
                train_x = train[0:, 0 : -1]
                train_y = train[0:, -1]
                valid_x = validation[0:, 0 : -1]
                valid_y = validation[0:, -1]
                valid_y = np.matrix(valid_y).transpose()
                # print valid_x.shape,'test_xshape'
                # print valid_y.shape,'test_yshape'
                # print valid_x.shape,'train_xshape'
                # print valid_y.shape,'train_yshape'
                w_c = w0
                train_m, train_n = train_x.shape  
                valid_m, valid_n = valid_x.shape
                for j in range(1,100000):
                        a_c = 1.0/ (j+ 1)
                        gradient = 1.0/train_m *((train_x.T * train_x) * w_c - train_x.T * train_y) + lamb/train_m * w_c
                        w_n = w_c - alpha*a_c* gradient
                        mm = w_n - w_c
                        sum_m = np.sum(np.square(mm))
                        sqrt = sum_m ** 0.5
                        # print sqrt
                        w_c = w_n
                        if k % 100 == 0:
                            print "try " + str(k) +"times"
                            # print "gradient"+ str(gradient)
                            print sqrt,'sqrt'
                        if sqrt < 0.2:
                            train_err = ((train_y - train_x * w_c).T * (train_y - train_x * w_c))/2.0/train_m 
                            train_err = np.asarray(train_err)
                            # print train_err[0][0], 'train_err'
                            # print type(train_err),'train_err_type'
                            train_errs[lamb][alpha].append(train_err[0][0])
                            # print m_test
                            valid_err = ((valid_y - valid_x * w_c).T * (valid_y - valid_x * w_c))/2.0/valid_m 
                            valid_err = np.asarray(valid_err)
                            # print valid_err.shape,'test_err shape'
                            # print valid_err[0][0], 'test_err'
                            valid_errs[lamb][alpha].append(valid_err[0][0]) 
                            break
            valid_err_mean[lamb][alpha] = np.mean(valid_errs[lamb][alpha])
            train_err_mean[lamb][alpha] = np.mean(train_errs[lamb][alpha])
    #lamb_o = min(test_err_mean,key = test_err_mean.get)
    #print lamb_o
    # print valid_err_mean,'test_err_mean'
    # print train_errs,'train_errs'    
    # print valid_errs,'valid_errs'
    minn = sys.maxint
    for k,v in valid_err_mean.iteritems():
        for k2,v2 in v.iteritems():
            print k,k2,v2
            if v2 < minn:
                minn = v2
                minn_k1 = k # best lambda
                minn_k2 = k2 #best alpha
                
    # print minn,minn_k1,minn_k2
    valid_err_mean_min = minn
    lamb = minn_k1
    alpha = minn_k2
    
    return  train_err_mean, valid_err_mean, train_errs, valid_errs,lamb,alpha,valid_err_mean_min
def ridgeGradientDescentWhole(training,lamb,alpha):
        training = np.asmatrix(training)
        train_x = training[0:, 0 : -1]
        train_y = training[0:, -1]
        #train_y = np.matrix(train_y).transpose()
        train_m, train_n = train_x.shape
        w_c = [0]*train_n
        w_c = np.asmatrix(w_c).T
        train_err_list = []
        for j in range(1,100000):
            a_c = 1.0 / (j+ 1)
            gradient = 1.0/train_m *((train_x.T * train_x) * w_c - train_x.T * train_y) + lamb/train_m * w_c
            w_n = w_c - alpha*a_c* gradient
            mm = w_n - w_c
            sum_m = np.sum(np.square(mm))
            sqrt = sum_m ** 0.5
            w_c = w_n
            train_err = ((train_y - train_x * w_c).T * (train_y - train_x * w_c))/2.0/train_m
            train_err = np.asarray(train_err)[0][0] 
            train_err_list.append(train_err)
            if j % 100 == 0:
                print "try " + str(j) +"times"
            # print "gradient"+ str(gradient)
                # print sqrt,'sqrt'
            if sqrt < 0.2:
                train_err_last = train_err
                wo = w_c
                break
        # print train_err_last,'training err on the whole data set'
        return wo,train_err_list,train_err_last
def noRidgeGradientDescentWhole(train,alpha):
    train = np.asmatrix(train)
    train_x = train[0:, 0 : -1]
    train_y = train[0:, -1]
    train_m, train_n = train_x.shape
    w_c = [0]*train_n
    w_c = np.asmatrix(w_c).T
    train_err_list = []
    for j in range(1,100000):
        a_c = 1.0 / (j+ 1)
        gradient = 1.0/train_m *((train_x.T * train_x) * w_c - train_x.T * train_y) 
        w_n = w_c - alpha*a_c* gradient
        mm = w_n - w_c
        sum_m = np.sum(np.square(mm))
        sqrt = sum_m ** 0.5
        w_c = w_n
        train_err = ((train_y - train_x * w_c).T * (train_y - train_x * w_c))/2.0/train_m 
        # + lamb/2.0/train_m * w_c.T*w_c
        train_err = np.asarray(train_err)[0][0]
        train_err_list.append(train_err)
        if j % 100 == 0:
            print "try " + str(j) +"times"
        # print "gradient"+ str(gradient)
            # print sqrt,'sqrt'
        if sqrt < 0.2:
            wo = w_c
            train_err_last = train_err
            break
    # print train_err_last,'training err on the whole data set'
    return wo,train_err_list,train_err_last
def touchTest(test,w_c):
    test_x = test[0:, 0 : -1]
    test_y = test[0:, -1]
    test_y = np.matrix(test_y).transpose()
    m_test, n_test = test_x.shape
    test_err = ((test_y - test_x * w_c).T * (test_y - test_x * w_c))/2.0/m_test 
    test_err = np.array(test_err)     
    return test_err[0][0]
def crossValidationNoRigeGradient(training,k):
   # print normalized_data
    n,m = training.shape
    #print n
    numPerFold = n/k
    # print numPerFold,'numPerFold'
    w0 = np.array([0.00]*(m-1))
    w0 = np.asmatrix(w0).T
    train_errs = defaultdict(list)
    valid_errs = defaultdict(list)
    #w = defaultdict(list)
    train_err_mean = defaultdict(list)
    valid_err_mean = defaultdict()
    
    # alphas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1 ,10, 100]
    alphas = [0.001,0.01,0.1,1,10]
    alphas = [0.1,1,10]
    for alpha in alphas:
        print alpha,'alpha'
        for i in range(k):
            # print i
            # print i * numPerFold,'i*numPerFold'
            validation = training[i * numPerFold: (i+1) * numPerFold, 0: ]
            # print validation.shape,'test.shape'
            rr =range(i * numPerFold , (i+1) * numPerFold + 1)
            train = np.delete(training, rr, axis=0)
            train = np.asmatrix(train)
            train_x = train[0:, 0 : -1]
            train_y = train[0:, -1]
            valid_x = validation[0:, 0 : -1]
            valid_y = validation[0:, -1]
            valid_y = np.matrix(valid_y).transpose()
            # print valid_x.shape,'test_xshape'
            # print valid_y.shape,'test_yshape'
            # print valid_x.shape,'train_xshape'
            # print valid_y.shape,'train_yshape'
            w_c = w0
            train_m, train_n = train_x.shape  
            valid_m, valid_n = valid_x.shape
            for j in range(1,100000):
                    a_c = 1.0/ (j+ 1)
                    gradient = 1.0/train_m *((train_x.T * train_x) * w_c - train_x.T * train_y)
                    w_n = w_c - alpha*a_c* gradient
                    mm = w_n - w_c
                    sum_m = np.sum(np.square(mm))
                    sqrt = sum_m ** 0.5
                    w_c = w_n
                    if k % 100 == 0:
                        print "try " + str(k) +"times"
                        # print "gradient"+ str(gradient)
                        # print sqrt,'sqrt'
                    if sqrt < 0.2:
                        train_err = ((train_y - train_x * w_c).T * (train_y - train_x * w_c))/2.0/train_m 
                        # + lamb/2.0/train_m * w_c.T*w_c
                        train_err = np.asarray(train_err)
                        # print train_err[0][0], 'train_err'
                        # print type(train_err),'train_err_type'
                        train_errs[alpha].append(train_err[0][0])
                        # print m_test
                        valid_err = ((valid_y - valid_x * w_c).T * (valid_y - valid_x * w_c))/2.0/valid_m 
                        #+ lamb/2.0/m_test * w_c.T*w_c
                        valid_err = np.asarray(valid_err)
                        # print valid_err.shape,'test_err shape'
                        # print valid_err[0][0], 'test_err'
                        valid_errs[alpha].append(valid_err[0][0]) 
                        break
            valid_err_mean[alpha] = np.mean(valid_errs[alpha])
            train_err_mean[alpha] = np.mean(train_errs[alpha])
    alpha_o = min(valid_err_mean,key = valid_err_mean.get)
    valid_err_mean_min = valid_err_mean[alpha_o]
    # print alpha_o,'alpha_o'
    # print valid_err_mean,'valid_err_mean'
    # print train_errs,'train_errs'    
    # print valid_errs,'valid_errs'
    
    return  train_err_mean, valid_err_mean, train_errs, valid_errs,alpha_o,valid_err_mean_min
def getDataMatrixPart2(file):
    with open(file,'rb') as csvFile:
            data = csv.reader(csvFile, delimiter = ',')
            data = list(data)
            data = np.asanyarray(data)
            data = data[1:,:]
            n, m = data.shape
            dataa =np.zeros((n,m-1))
            dataa[0:,0:-1] = data[0:,2:]
            dataa[0:,-1] = data[0:,0]
    return dataa
def plot3D(dic):
    fig = plt.figure()
    ax = Axes3D(fig)  
    x = []
    y = []
    z = []
    for k1, v1 in dic.iteritems():
        for k2, v2 in v1.iteritems():
            x.append(k1)
            y.append(k2)
            z.append(v2)
    ax.scatter(x,y,z,c = 'r', marker = 'o')
    ax.set_xlabel('lambda')
    ax.set_ylabel('alpha')
    ax.set_zlabel('validation error')
    ax.set_title('mean validation error based on alpha and lambda')
    plt.show()
def plot2d(dic):
    x=[]
    y=[]
    for k,v in dic.iteritems():
        x.append(k)
        y.append(v)
    fig = plt.figure()
    plt.scatter(x,y)
    plt.xlabel('alpha')
    plt.ylabel('validation error')
    plt.title('validation error based on different alpha ')
    plt.show()
def plotCrossEffect(training,test):
    train_err_l = []
    test_err_l = []
    valid_err_l = []
    x = range(2,11)
    for kk in range(2,11):
        print 'using ' + str(kk) + 'fold'
        train_err_mean, valid_err_mean, train_errs, valid_errs,lamb,alpha,valid_err_mean_min = crossvalidationRidgeGradient(training, kk)
        wo,train_err_list,train_err = ridgeGradientDescentWhole(training,lamb,alpha) 
        test_err = touchTest(test,wo)
        train_err_l.append(train_err)
        test_err_l.append(test_err)
        valid_err_l.append(valid_err_mean_min)
    fig = plt.figure()
    plt.plot(x,train_err_l,'r', x, valid_err_l,'bs',x,test_err_l,'g^')
    plt.legend(['train_err','validation error','test_err'])
    plt.title('crossvalidationeffect of rigdge gradient descent')
    plt.show()
def plotCrossEffect2(training,test):
    train_err_l = []
    test_err_l = []
    valid_err_l = []
    x = range(2,11)
    for kk in range(2,11):
        print 'using ' + str(kk) + 'fold'
        train_err_mean, valid_err_mean, train_errs, valid_errs,alpha,valid_err_mean_min = crossValidationNoRigeGradient(training, kk)
        wo,train_err_list,train_err = noRidgeGradientDescentWhole(training,alpha) 
        test_err = touchTest(test,wo)
        train_err_l.append(train_err)
        test_err_l.append(test_err)
        valid_err_l.append(valid_err_mean_min)
    fig = plt.figure()
    plt.plot(x,train_err_l,'r', x, valid_err_l,'bs',x,test_err_l,'g^')
    plt.legend(['train_err','validation error','test_err'])
    plt.title('crossvalidationeffect of no rigdge gradient descent')
    plt.show()
def plotIterationResult(train_err_list):
    x = range(1,len(train_err_list) + 1)
    fig = plt.figure()
    plt.plot(x,train_err_list)
    plt.xlabel('iterations')
    plt.ylabel('training error')
    plt.show()
def main():
    # choose = input("You wana test part 1 or part 2 using gradient descent? 1 or 2")
    
    file2 = "final.csv"
    file1 = "OnlineNewsPopularity.csv"
    data = read(file1)
    # if choose == 2:
    #     data = getDataMatrixPart2(file2)
    # elif choose == 1:
    #     data = read(file1)
    # else:
    #     print "type 1 or 2, you can not type others"
    #     return
    
    #normalize data
    print "normalize data"
    data = nomalize(data)
    print data.shape
    training, test = splitDataSet(data)
    print training.shape

    #plotting errors
    # plotCrossEffect(training,test)
    # plotCrossEffect2(training,test)



    # print training.shape
    # print test.shape
    k = 5
    print "ridge"
    #ridge
    train_err_mean, valid_err_mean, train_errs, valid_errs,lamb,alpha,valid_err_mean_min = crossvalidationRidgeGradient(training, k)
    wo,train_err_list0,train_err0 = ridgeGradientDescentWhole(training,lamb,alpha) 
    test_err = touchTest(test,wo)
    print test_err,'test_err ridge'
    print train_err0,'training error on the whole set on no ridge '
    print valid_err_mean,'validation error mean ridge'
    print valid_errs,'validationerros  ridge'
    print lamb,alpha,'best lamb,alpha ridge'
    plot3D(valid_err_mean)
    plotIterationResult(train_err_list0)
    print 'no ridge'
    #no ridge
    train_err_mean, valid_err_mean, train_errs, valid_errs,alpha_o,valid_err_mean_min = crossValidationNoRigeGradient(training,k)
    w1,train_err_list1,train_err1 = noRidgeGradientDescentWhole(training,alpha_o) 
    test_err = touchTest(test,w1)
    print test_err, 'test_err on no ridge'
    print train_err1,'training error on the whole set on no ridge '
    print valid_err_mean,'validation error mean no ridge'
    print valid_errs,'validationerros no ridge'
    print alpha_o,'best alpha  no ridge'

    plotIterationResult(train_err_list1)
    plot2d(valid_err_mean)



if __name__ == '__main__':
    main()