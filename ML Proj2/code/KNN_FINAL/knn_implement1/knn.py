#__author__ = 'Admin'

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
    words =[word.strip(".__EOS!,?") for word in words]
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

shuffle(CVS_DATA)   # Randomize data

CSV_DATA=[]

with open("ml_dataset_test_in.csv",'r') as testinput:
    testreader = csv.reader(testinput)

    for row in testreader:
        #print row[1]
        if row[1]!='':                   # Remove empty lines from file
            CSV_DATA.append(row[1])
CSV_DATA.pop(0)
testoutput = open("preet.csv",'w')
columns = ['Id','Prediction']
testoutput.write('\t'.join(columns) + '\n')
#Getting all the stop words to be eliminated
stop_words = get_stop_words('english')
#Dividing the Training and Validation Data (For the moment I am not doing Cross Validation, Once I get working Algorithm, its easy to implement
num_train_examples  = len(CVS_DATA)
train_examples = CVS_DATA[:num_train_examples]
num_test_examples = len(CSV_DATA)
valid_examples = CSV_DATA[:]
output ={}
auth =[]
movie = []
music =[]
interview =[]
frequency = dict()
au = open("au.txt",'r')
for i in au:
    s = i.strip()
    auth.append(s)
for j in auth:
    frequency[j] = frequency.get(j,0) + 1

mo = open("mov.txt",'r')
for i in mo:
    s = i.strip()
    movie.append(s)
for j in movie:
    frequency[j] = frequency.get(j,0) + 1

mu = open("musics.txt",'r')
for i in mu :
    s = i.strip()
    music.append(s)
for j in music:
    frequency[j] = frequency.get(j,0) + 1

inter = open("interview.txt",'r')
for i in inter:
    s = i.strip()
    interview.append(s)
for j in interview:
    frequency[j] = frequency.get(j,0) + 1

#print len(CSV_DATA)


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
trainfeatures.append((auth,'0'))
trainfeatures.append((movie,'1'))
trainfeatures.append((music,'2'))
trainfeatures.append((interview,'3'))
    #print trainfeatures
#print "FREQUENCY OF DIFFERENT WORDS SEEN IN TRAINING EXAMPLES"
#print frequency

# Evaluation of Test Set
m =0
valid_wrong_classif = 0 # Wrong Classifications on Valid set are represented by this variable
for line in valid_examples:
    validwords = get_words(line)
    results =[]
    #going over every train example and computing similarity
    for i,(trainwords,trainlabel) in enumerate(trainfeatures):
        # find all words in common between the two sentences
        common_words = [x for x in trainwords if x in validwords ]
        # Accumulating score for all possible overlaps
        score = 0.0
        for word in common_words:
            try:
                score += log(num_train_examples/frequency[word])
            except:
                print "Word Cannot be found!!!"
        results.append((score, trainlabel))
    results.sort(reverse=True)
    #print "SCORES"
    #print results
    ## We need to do cross-validation to choose the value of k also, but time being i am just considering k = 5, 5 nearest neighbour
    #classifier
    classify ={}
    n = 10
    top_labels = [x[1] for x in results[:n]]
    #print "FIRST %d top labels for given example in VALIDATION SET"%n
    #print top_labels
    classify['author'] = top_labels.count('0')
    classify['movie'] = top_labels.count('1')
    classify['music'] = top_labels.count('2')
    classify['interviews']= top_labels.count('3')
    #print"Dictionary representing votes to different classes"
    #print classify

    predicted_class = sorted(classify.items(),key = operator.itemgetter(1),reverse=True)
    #print predicted_class
    prediction = predicted_class[0]
    if prediction[0] == 'author':
        print "The predicted class is author"
        yhat='0'
    elif prediction[0] =='movie':
        print "The predicted class is movie"
        yhat = '1'
    elif prediction[0] == 'music':
        print "The predicted class is music"
        yhat ='2'
    else:
        print "The predicted class is interviews"
        yhat = '3'
    print yhat
    output['Id'] = m
    #output['Interview']=line
    output['Prediction'] = yhat
    values = map(lambda col : output.get(col,''),columns)
    values = [str(v) for v in values]
    testoutput.write('\t'.join(values) + '\n')
    m += 1




















