'''
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
import sys

import data_extractor
import defines
import svm
import CrossVal
# Set logger

logging.basicConfig(filename='output.log',level=logging.DEBUG)

#logging.info('This is an info log')
#logging.warning('This is a warning log')
#logging.error('This is an error log')

#*******************************************************************************
# DATA EXTRACTION
TRAIN_DATA = data_extractor.get_data(defines.DATA_TRAIN_CSV_FILE)
TEST_DATA = data_extractor.get_data(defines.DATA_TEST_CSV_FILE)

#*******************************************************************************
# STOP WORDS FILTER
#logging.info('Prepare Data')
#nb_lib.nb_lib_prepare(TRAIN_DATA)

#*******************************************************************************
# FEATURE SELECTION
logging.info('Feature selection')
#feature_select.get_selected_features(TRAIN_DATA)
#*******************************************************************************

logging.info('SVM')

svm.run_svm(TRAIN_DATA, TEST_DATA)
logging.info ('Cross Validation')

#CrossVal.run_crossval(TRAIN_DATA)

print ("done!")
sys.exit()
