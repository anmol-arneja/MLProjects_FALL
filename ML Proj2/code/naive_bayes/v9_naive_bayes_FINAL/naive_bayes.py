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
from random import shuffle
import os
import numpy as np
import re
import defines
import logging
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

def run_nb(TRAIN_DATA, TEST_DATA):

    logging.info('\n\n =================== START ===================')
    logging.info('TRAIN Data length = %d'%(len(TRAIN_DATA)))
    logging.info('TEST Data length  = %d'%(len(TEST_DATA)))
    logging.info('Number of features= %d'%(len(defines.SELECTED_FEATURES)))

    print('get counters!')
    [COUNTERS, all_c, real_labels] = get_nb_counter(TRAIN_DATA, TEST_DATA)

    print ('train naive bayes algorithm')
    [cond_prob, priors] = train_nb(TRAIN_DATA, COUNTERS)

    print ('Get predictions')
    predictions = predict_nb(cond_prob, priors, all_c)

    print ('Find accuracy')

    if defines.USE_TEST_DATA == 0:
        miss_count = []
        samples_num = len(predictions)
        mismatch = 0
        logging.info('\n\n =================== MISMATCH REPORT ===================')
        for i in range(0, samples_num):
            if (predictions[i] != real_labels[i]):
                miss_count.append((defines.LABEL_DIC[predictions[i]], defines.LABEL_DIC[real_labels[i]]))
                mismatch += 1
                logging.info('M:[%d] - predicted: %s, but labeled as: %s' %(i, defines.LABEL_DIC[predictions[i]], defines.LABEL_DIC[real_labels[i]]))
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

def predict_nb(cond_prob, priors, all_c):
    
    predictions = []

    sample_index = 0
    logging.info('\n\n =================== SCORE REPORT ===================')
    for sample_counters in all_c:
        score = np.zeros(4)
        for class_index in range(0,4):
            score[class_index] = np.log(priors[class_index])
            feature_index = 0
            for word_count in sample_counters:
                score[class_index] += word_count * np.log(cond_prob[class_index][feature_index])
                feature_index += 1

        chosen_label = np.argmax(score)

        BIAS_DIFF = 0.3
        if chosen_label == defines.LABEL_AUTHOR:
            if (score[defines.LABEL_INTERVIEW] > score[defines.LABEL_MOVIES]) and (score[defines.LABEL_INTERVIEW] > score[defines.LABEL_MUSIC]):
                 score[defines.LABEL_AUTHOR] = score[defines.LABEL_AUTHOR] - (BIAS_DIFF)
                 score[defines.LABEL_INTERVIEW] = score[defines.LABEL_INTERVIEW] - (BIAS_DIFF/2)
                 chosen_label = np.argmax(score)
                 if chosen_label != defines.LABEL_AUTHOR:
                     logging.info('[%d]hacking from AUTHOR --> %s' %(sample_index, defines.LABEL_DIC[chosen_label]))

        logging.info('[%d]: Predicted class %s with score: %f' %(sample_index,defines.LABEL_DIC[chosen_label], score[chosen_label]))
        logging.info(score)
        logging.info('--------------------------')
        predictions.append(chosen_label)
        sample_index += 1

    print ('done making predictions')
    return predictions




def get_nb_counter(CSV_DATA, TEST_DATA):
    samples = [ row[0] for row in CSV_DATA ]
    labels  = [ row[1] for row in CSV_DATA ]

    samples_test = [ row[0] for row in TEST_DATA ]
    if defines.USE_TEST_DATA == 0:
        labels_test  = [ row[1] for row in TEST_DATA ]
        labels_test = [int(i) for i in labels_test]
    else:
        labels_test = []

    
    labels = [int(i) for i in labels]
    labels = np.asarray(labels)

    #Vectorize and encode selected features/words
    data_vect = CountVectorizer(vocabulary = defines.SELECTED_FEATURES,
                                 min_df = 1,
                                 strip_accents = 'ascii',
                                 analyzer = 'word',
                                 stop_words = 'english' )


    data_fit = data_vect.fit_transform(samples)

    features = data_vect.get_feature_names()

    # counters for each sample
    all_c = data_fit.toarray()

    # Get counters for each class: number of repetitions for each
    # selected word in each calss across all samples
    author_c    = np.sum(all_c[labels == defines.LABEL_AUTHOR],     axis=0)
    movies_c    = np.sum(all_c[labels == defines.LABEL_MOVIES],     axis=0)
    music_c     = np.sum(all_c[labels == defines.LABEL_MUSIC],      axis=0)
    interview_c = np.sum(all_c[labels == defines.LABEL_INTERVIEW],  axis=0)

    test_data_fit = data_vect.fit_transform(samples_test)

    all_c_teset =  test_data_fit.toarray()

    print('done preparing data')

    

    return (np.vstack ((author_c, movies_c, music_c, interview_c)), all_c_teset, labels_test)

def train_nb(TRAIN_DATA, FEATURE_COUNTS):
    labels  = [ row[1] for row in TRAIN_DATA ]
    labels = [int(i) for i in labels]


    N = len(TRAIN_DATA)

    print('get Nc counters')
    priors = [labels.count(defines.LABEL_AUTHOR)/N,
              labels.count(defines.LABEL_MOVIES)/N,
              labels.count(defines.LABEL_MUSIC)/N,
              labels.count(defines.LABEL_INTERVIEW)/N
              ]

    class_count_sum = np.sum(FEATURE_COUNTS, axis= 1)

    #calculate posterior probabilities
    cond_prob = np.zeros([4,len(FEATURE_COUNTS[0])])
    for class_index in range(0,4):
        f_index = 0
        for feature in FEATURE_COUNTS[class_index]:
            cond_prob[class_index][f_index] = float((feature+1))/float((class_count_sum[class_index] + len(FEATURE_COUNTS[0])))
            f_index += 1

    return (cond_prob,priors)


