'''
COMP-598 - Applied Machine Learning 
Project 2 - Classification

data_extractor.py = This file performs data extraction for a given 
                    CSV file
@authors: Sandra Maria Nawar
          Timardeep Kaur
          Daniel Galeano
October 20th, 2015
'''

#*******************************************************************************
# IMPORT LIBRARIES AND TOOLS
import logging
import csv
from random import shuffle
import os
import numpy as np

#********************************************************************************************************
# EXTRACT DATA FROM CSV FILE
def get_data(cvs_file):
    CVS_DATA = []
    print ("Reading data from %s" %(cvs_file))
    csv_file_path = os.getcwd() + '/' + cvs_file
    #with open(csv_file_path,'r',encoding= 'cp850') as f:
    with open(csv_file_path,'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 0:                   # Remove empty lines from file
                CVS_DATA.append(row[1:])        # Ignore the Id column

    CVS_DATA.pop(0)                             # Remove first row with CSV titles

    shuffle(CVS_DATA)                           # Randomize data

    NP_ARRAY = np.asanyarray(CVS_DATA)          # Convert to numpy array
    return NP_ARRAY