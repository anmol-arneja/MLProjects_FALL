'''
COMP-598 - Applied Machine Learning 
Project 2 - Classification

defines.py = Contains global definitions
@authors: Sandra Maria Nawar
          Timardeep Kaur
          Daniel Galeano
October 20th, 2015
'''

#*******************************************************************************
#*******************************************************************************
import numpy as np
import selected_features
import data
from random import shuffle

# DEFINITIONS
DATA_TRAIN_CSV_FILE = "ml_dataset_train.csv"
DATA_TEST_CSV_FILE = "ml_dataset_test_in.csv"
#DATA_TRAIN_CSV_FILE = "train.csv"
#DATA_TEST_CSV_FILE = "test.csv"

#FEATURE_SELECTED = 5000

USE_TEST_DATA = 1

LABEL_AUTHOR    = 0
LABEL_MOVIES    = 1
LABEL_MUSIC     = 2
LABEL_INTERVIEW = 3

LABEL_DIC = ['author','movies','music','interview']

LABELS = np.array([LABEL_AUTHOR, LABEL_MOVIES, 
                   LABEL_MUSIC, LABEL_INTERVIEW])


# Build Selected Features
#select_2 = selected_features.RAW_SELECTED_FEATURES_2
#shuffle(select_2)
#select_temp = set(selected_features.RAW_SELECTED_FEATURES + select_2)
select_temp = data.LAST
select_temp = list(set(select_temp))
FEATURE_NUM = int(len(select_temp)) - 1
SELECTED_FEATURES = select_temp[:FEATURE_NUM - 1]
