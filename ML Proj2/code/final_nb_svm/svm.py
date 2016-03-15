'''
COMP-598 - Applied Machine Learning
Project 2 - Classification

naive_bayes.py = This file contains the algorithm for the naive bayes classifier

@authors: Sandra Maria Nawar
          Timardeep Kaur
          Daniel Galeano
October 20th, 2015
'''
#*******************************************************************************

import csv
import numpy as np
import defines
import logging
import random
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

def batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

def run_svm(TRAIN_DATA, TEST_DATA):

    logging.info('\n\n =================== START ===================')
    logging.info('TRAIN Data length = %d'%(len(TRAIN_DATA)))
    logging.info('TEST Data length  = %d'%(len(TEST_DATA)))
    logging.info('Number of features= %d'%(len(defines.SELECTED_FEATURES)))

    print('get counters!')
    [x_train, x_test, y_train, y_test] = get_counters(TRAIN_DATA, TEST_DATA)
    
    #clf = SVC( kernel='rbf', cache_size=1000)
    clf = linear_model.SGDClassifier(n_iter=50, penalty="elasticnet")  #loss='log, modified_huber' #penalty="elasticnet"
    #clf = MultinomialNB()
    #clf = KNeighborsClassifier(n_neighbors=3)

    shuffledRange = list(range(len(x_train)))

    n_iter = 100
    for n in range(n_iter):
        print('iteration: [%d]'%(n))
        random.shuffle(shuffledRange)
        shuffledX = [x_train[i] for i in shuffledRange]
        shuffledY = [y_train[i] for i in shuffledRange]
        for batch in batches(range(len(shuffledX)), 1000):
            clf.partial_fit(shuffledX[batch[0]:batch[-1]+1], shuffledY[batch[0]:batch[-1]+1], classes=np.unique(y_train))

    print('Training Classifier...')
    #clf.fit(x_train, y_train)
    
    predictions = []
    print('making predictions ...')
    for sample in x_test: 
        predictions.append(clf.predict([sample]))
    
    
    print ('Find accuracy')

    if defines.USE_TEST_DATA == 0:
        miss_count = []
        samples_num = len(predictions)
        mismatch = 0
        logging.info('\n\n =================== MISMATCH REPORT ===================')
        for i in range(0, samples_num):
            if (predictions[i] != y_test[i]):
                miss_count.append((defines.LABEL_DIC[predictions[i]], defines.LABEL_DIC[y_test[i]]))
                mismatch += 1
                logging.info('M:[%d] - predicted: %s, but labeled as: %s' %(i, defines.LABEL_DIC[predictions[i]], defines.LABEL_DIC[y_test[i]]))
                logging.info('----------------------------------------------------------\n')

        logging.info('\n\n =================== ACCURACY AND TUPLES COUNTERS REPORT ===================')
        logging.info('Total number of misses = %d' %(mismatch))
        # Accuracy
        accuracy = 100 * float(mismatch)/float(samples_num)
        print ('Accuracy = %f' %(100 - accuracy))
        logging.info('Accuracy = %f' %(100 - accuracy))

        # Tuples info
        predict_miss = Counter(elem[0] for elem in miss_count)
        real_miss = Counter(elem[1] for elem in miss_count)
        logging.info(' tuples: predict miss counters: ')
        logging.info(predict_miss)
        logging.info(' tuples: real miss counters: ')
        logging.info(real_miss)


    logging.info('\n\n =================== PREDICTIONS ===================')
    print (predictions)
    logging.info(predictions)

    print ('Export to CSV file')
    results = []
    for index in range(0,len(predictions)):
        tuple = (index, predictions[index])
        results.append(tuple)
    print ('results')
    print (results)
    logging.info(predictions)

    with open('results.csv','w') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(['id','prediction'])
        for row in results:
            csv_out.writerow(row)

    print('done exporting to CSV file')



def get_counters(TRAIN_DATA, TEST_DATA):
    samples = [ row[0] for row in TRAIN_DATA ]
    labels_train  = [ row[1] for row in TRAIN_DATA ]
    labels_train = [int(i) for i in labels_train]
    labels_train = np.asarray(labels_train)

    samples_test = [ row[0] for row in TEST_DATA ]
    if defines.USE_TEST_DATA == 0:
        labels_test  = [ row[1] for row in TEST_DATA ]
        labels_test = [int(i) for i in labels_test]
    else:
        labels_test = []

    
    

    #Vectorize and encode selected features/words
    data_vect = CountVectorizer(vocabulary = defines.SELECTED_FEATURES,
                                 min_df = 1,
                                 strip_accents = 'ascii',
                                 analyzer = 'word',
                                 stop_words = 'english' )


    data_fit = data_vect.fit_transform(samples)

    #features = data_vect.get_feature_names()

    # counters for each sample
    all_c_train = data_fit.toarray()


    test_data_fit = data_vect.fit_transform(samples_test)

    all_c_test =  test_data_fit.toarray()

    print('done preparing data')

    

    return (all_c_train, all_c_test, labels_train, labels_test)

