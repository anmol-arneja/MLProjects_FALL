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
import feature_select

#defining my inputs
clas = np.array([defines.LABEL_AUTHOR ,defines.LABEL_MOVIES ,defines.LABEL_MUSIC,defines.LABEL_INTERVIEW])
print clas
#add vocab string of words
#extract words that appear in each class most


FEATURE_SELECTED=["time","book","say","thought","more","don","over","into","people","thank","got","work","come","life","sort","actually","see","thanks","man","author"]


def NBAlg(clas, FEATURE_SELECTED):
     words=list(FEATURE_SELECTED)
     #words has to be list form
     for x, words in enumerate(words):
         N = len(words.split())         #Total number of words
         words.index("English") #Error
         for i in clas:
         #count number of words in each class???

      prior = /N

      # Concatenate text of all docs in each class

        for t in W:
            #count the number of words in each class
         T =
        condprob[t][c] = T / sum(T)+1

     return words,prior,condprob





def Classifier(clas, vocab, prior, condprob, data):
     W = vocab
     score_obtained = []
     score = []
     conprob =[]
     res = []
     for i in clas:                                 #for all classes
        score[i] = np.log(prior[i])
        for t in W:                                 #for all words
            score[i]+= np.log(conprob[t][i])        #sum log of each prob over classes and texts

     score_obtained.append(score)                   #store in matrix prob of being in each class
     max = clas[np.argmax(score_obtained)]          #Choose class that has highest prob
     res.append(max)

     return res

