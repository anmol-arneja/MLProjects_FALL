__author__ = 'Admin'

from math import sqrt,log
from nltk.stem import PorterStemmer
import csv
import re
import pandas
from stop_words import get_stop_words
from random import shuffle
import os
import operator

num = re.compile('[0-9]+')
def get_words(conversation):
    '''Here we are taking each conversation and
    trying to extarct words from conversation'''
    words = conversation.split()
    words = [word for word in words if len(word)>2 ]
    #eliminating all the stop words from list of words
    words = [word for word in words if word not in stop_words]
    #print len(words)
    words =[word.strip(".__EOS!?") for word in words]
    words = list(set(words))
    #print len(words)
    return words

#Reading the Training Data

CVS_DATA = []
#print ("Reading data from %s" %(cvs_file))
#csv_file_path = os.getcwd() + '/' + cvs_file
#with open(csv_file_path,'r',encoding= 'cp850') as f:
with open('ml_dataset_train.csv','r') as f:
    reader = csv.reader(f)
    for row in reader:
        #print row[1]
        if row[1]!='':                   # Remove empty lines from file
            CVS_DATA.append(row[1:])        # Ignore the Id column

CVS_DATA.pop(0)                             # Remove first row with CSV titles

shuffle(CVS_DATA)                           # Randomize data

#Getting all the stop words to be eliminated
stop_words = get_stop_words('english')
#Dividing the Training and Validation Data (For the moment I am not doing Cross Validation, Once I get working Algorithm, its easy to implement
num_train_examples  = 11
train_examples = CVS_DATA[:num_train_examples]

valid_examples = CVS_DATA[num_train_examples:20]


# Initializing the stemmer objects for stemming
stemmer = PorterStemmer()


'''computing frequency of words in training set, we will want to give less importance to commonly used words'''

frequency = dict()
trainfeatures =[]
for line,label in train_examples: # for every conversation and its label in training examples
    w = get_words(line)
    for i in w:
        frequency[i] =  frequency.get(i,0) + 1
    trainfeatures.append((w, label))
    print trainfeatures

# Evaluation of Test Set
valid_wrong_classif = 0 # Wrong Classifications on Valid set are represented by this variable
for line,label in valid_examples:
    validwords = get_words(line)
    results =[]
    #going over every train example and computing similarity
    for i,(trainwords,trainlabel) in enumerate(trainfeatures):
        # find all words in common between the two sentences
        common_words = [x for x in trainwords if x in validwords ]
        # Accumulating score for all possible overlaps
        score = 0.0
        for word in common_words:
            score += log(num_train_examples/frequency[word])
        results.append((score, trainlabel))
    results.sort(reverse=True)
    print results
    ## We need to do cross-validation to choose the value of k also, but time being i am just considering k = 5, 5 nearest neighbour
    #classifier
    classify ={}
    top_labels = [x[1] for x in results[:5]]
    print top_labels
    classify['author'] = top_labels.count('0')
    classify['movie'] = top_labels.count('1')
    classify['music'] = top_labels.count('2')
    classify['interviews']= top_labels.count('3')
    print classify

    predicted_class = sorted(classify.items(),key = operator.itemgetter(1),reverse=True)
    print predicted_class
    prediction = predicted_class[0]

    if prediction[0] == 'author':
        print "The predicted class is author"
    elif prediction[0] =='movie':
        print "The predicted class is movie"
    elif prediction[0] == 'music':
        print "The predicted class is music"
    else:
        print "The predicted class is interviews"











