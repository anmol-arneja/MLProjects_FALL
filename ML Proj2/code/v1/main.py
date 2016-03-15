﻿'''
COMP-598 - Applied Machine Learning 
Project 2 - Classification

main.py = This file is the entry point for the algorithms.
          Please refer to the README File to understand how
          to configure the running modes  
@authors: Sandra Maria Nawar
          Timardeep Kaur
          Daniel Galeano
October 20th, 2015
'''

#*******************************************************************************
# IMPORT LIBRARIES AND TOOLS
import logging
import numpy as np
import sys
import csv

import data_extractor

# Set logger
logging.basicConfig(filename='output.log',level=logging.DEBUG)

#logging.info('This is an info log')
#logging.warning('This is a warning log')
#logging.error('This is an error log')

#*******************************************************************************
# DEFINITIONS
DATA_TRAIN_CSV_FILE = "ml_dataset_train.csv"
DATA_TEST_CSV_FILE = "ml_dataset_test_in.csv"

#*******************************************************************************
# DATA EXTRACTION
TRAIN_DATA = data_extractor.get_data(DATA_TRAIN_CSV_FILE)
TEST_DATA = data_extractor.get_data(DATA_TEST_CSV_FILE)

#*******************************************************************************
# STOP WORDS FILTER
logging.info('Stop Words Filter')
#TODO: Add function here to call stop word filter!

#*******************************************************************************
# FEATURE SELECTION
logging.info('Feature selection')
#TODO: Add function here to call feature selector!

#*******************************************************************************
# DERIVATION OF NAIVE BAYES CLASSIFIER 
logging.info('Naive Bayes Classifier')
#TODO: Add function here to call naive bayes classifier!

print ("done!")
sys.exit()
