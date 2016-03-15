from __future__ import division
from graphlab import SArray
from graphlab import SFrame
import graphlab
import csv
import decimal
import operator
from operator import  itemgetter
import random
import numpy
from math import log
# Pre Processing the data given to us
input_data = open('ml_dataset_train.csv','r')
output_data = open('processed_ml_dataset_train.csv','w')
writer = csv.writer(output_data)
for row in csv.reader(input_data):
    if row[0]!= '':
        writer.writerow(row)
input_data.close()
output_data.close()

# Loading the Training Data
data = graphlab.SFrame.read_csv('processed_ml_dataset_train.csv')
#few_lines = data.head()
#print few_lines
print "Loading our Training Dataset"
tot_lines = len(data)
print "Dataset contains %d interview excerpts"%tot_lines
print "Done with loading of Training Dataset"

#Splitting the Dataset into Training and Test Data
valid_wrong_classif = 0
n = 50000
train_examples = data[0:n]
test_examples = data[n:]
num_test_examples = len(test_examples)
train_examples['word_count']=graphlab.text_analytics.count_words(train_examples['Interview'])
print train_examples['word_count']

tfidf_train = graphlab.text_analytics.tf_idf(train_examples['word_count'])

train_examples['tfidf'] = tfidf_train['docs']


for i in range(len(test_examples)):
    dictList =[]
    classify ={}
    preet = []
    Creet =[]
    frequency = {}
    data1 ={}
    a = test_examples[i]['Interview'].split()

    for j in a:
        frequency[j] = frequency.get(j,0) + 1
        data1[j] = log (num_test_examples/frequency[j])
    preet.append(frequency)
    #print data1

    '''b =SArray(preet)
    tfidf  = (graphlab.text_analytics.tf_idf(b))
    data1['tfidf'] = tfidf['docs']
    print data1
    #print train_examples['tfidf'][1]
    #print data1['tfidf'][0]'''

    for k in range(len(train_examples)):

        a = graphlab.distances.cosine(data1 ,train_examples['tfidf'][k])
        dictList.append((a,int(train_examples[k]['Prediction'])))
    dictList.sort(key = itemgetter(0))

    top_labels = [x[1] for x in dictList[0:5]]
    #print top_labels
    classify['author'] = top_labels.count(0)
    classify['movie'] = top_labels.count(1)
    classify['music'] = top_labels.count(2)
    classify['interviews']= top_labels.count(3)

    predicted_class = sorted(classify.items(),key = operator.itemgetter(1),reverse=True)
    prediction = predicted_class[0]
    #print prediction
    if prediction[0] == 'author':
        yhat=0
        print yhat
        label = train_examples[k]['Prediction']
        print label
    elif prediction[0] =='movie':

        yhat = 1
        print yhat
        label = train_examples[k]['Prediction']
        print label

    elif prediction[0] == 'music':

        yhat = 2
        print yhat
        label = train_examples[k]['Prediction']
        print label

    else:

        yhat = 3
        print yhat
        label = train_examples[k]['Prediction']
        print label

    if yhat!= label:
        valid_wrong_classif += 1
print "The toatal number of errors are %d"%valid_wrong_classif
