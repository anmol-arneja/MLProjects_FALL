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

#defining my inputs
clas = np.array([0,1,2,3])

#add vocab string of words
#extract words that appear in each class most
L0 = ["English", "Book"]
L1 = ["Film", "Actor"]
L2 = ["Song", "Lyrics"]
L3 = ["Job", "Career"]

#matrix containing all the words

words =[L0,L1,L2,L3]

print words

def NBAlg(clas, words):
     words=list(words)
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


