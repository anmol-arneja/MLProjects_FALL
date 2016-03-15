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
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer


def run_nb(TRAIN_DATA):

    print('get counters!')
    [COUNTERS, all_c] = get_nb_counter(TRAIN_DATA)

    print 'train naive bayes algorithm'
    [cond_prob, priors] = train_nb(TRAIN_DATA, COUNTERS)

    print 'Get predictions'
    predict_nb(cond_prob, priors, all_c)

    print 'MisClassification Error'
    misclass_error(chosen_label)
    print('done')

def predict_nb(cond_prob, priors, all_c):
    score = np.zeros(4)

    for samples in range(0,len(all_c)):

        for class_index in range(0,4):
            feature_index = 0
            score[class_index] = np.log(priors[class_index])
            for feature_count in all_c:                     #all columns ??
              #print cond_prob[class_index][feature_index]
              score[class_index] += feature_count[samples] * np.log(cond_prob[class_index][feature_index])
              chosen_label = np.argmax(score[samples])
              print ('class is: %d with score: %f'%(chosen_label, score[chosen_label]))
              feature_index += 1

    return chosen_label

#def misclass_error(chosen_label):





def get_nb_counter(CSV_DATA):
    samples = [ row[0] for row in CSV_DATA ]
    labels  = [ row[1] for row in CSV_DATA ]

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


    print('done preparing data')

    return (np.vstack ((author_c, movies_c, music_c, interview_c)), all_c)

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



