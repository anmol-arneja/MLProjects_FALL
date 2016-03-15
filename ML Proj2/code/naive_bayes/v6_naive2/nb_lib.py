'''
COMP-598 - Applied Machine Learning 
Project 2 - Classification

feature_select.py = This file performs feature selection for a given data
@authors: Sandra Maria Nawar
          Timardeep Kaur
          Daniel Galeano
October 20th, 2015
'''

#*******************************************************************************
# IMPORT LIBRARIES AND TOOLS
#*******************************************************************************
import logging
import numpy as np
#from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

import defines

def nb_lib_prepare(CSV_DATA):
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
    
    all_c = data_fit.toarray()
    
    # Get counters for each class: number of repetitions for each 
    # selected word in each calss across all samples
    author_c    = np.sum(all_c[labels == defines.LABEL_AUTHOR],     axis=0)
    movies_c    = np.sum(all_c[labels == defines.LABEL_MOVIES],     axis=0)
    music_c     = np.sum(all_c[labels == defines.LABEL_MUSIC],      axis=0)
    interview_c = np.sum(all_c[labels == defines.LABEL_INTERVIEW],  axis=0)

    print('done preparing data')
    
    
    
    
    
    
    
