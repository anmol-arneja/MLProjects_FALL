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

# DEFINITIONS
#DATA_TRAIN_CSV_FILE = "train.csv"
#DATA_TEST_CSV_FILE = "test.csv"
#DATA_TRAIN_CSV_FILE = "ml_dataset_train.csv"
#DATA_TEST_CSV_FILE = "ml_dataset_test_in.csv"
DATA_TRAIN_CSV_FILE = "train.csv"
DATA_TEST_CSV_FILE = "test.csv"

LABEL_AUTHOR    = 0
LABEL_MOVIES    = 1
LABEL_MUSIC     = 2
LABEL_INTERVIEW = 3

FEATURE_SELECTED = 200

SELECTED_FEATURES = ['book','life','story','write','read',
                     'film','movie','people','character','scene','actor',
                     'music','song','album','play','songs','band',
                     'say','talk','thank','talking','ask','issue']


LABELS = np.array([LABEL_AUTHOR, LABEL_MOVIES, 
                   LABEL_MUSIC, LABEL_INTERVIEW])

