import graphlab
import csv
import operator
from operator import  itemgetter
import random

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
print "Done with loading of Dataset"

#Splitting the Dataset into Training and Test Data
valid_wrong_classif = 0
n = 52000
train_examples = data[0:n]
test_examples = data[n:]
train_examples['word_count']=graphlab.text_analytics.count_words(train_examples['Interview'])
test_examples['word_count'] = graphlab.text_analytics.count_words(test_examples['Interview'])
tfidf_train = graphlab.text_analytics.tf_idf(train_examples['word_count'])
#print tfidf
train_examples['tfidf'] = tfidf_train['docs']
tfidf_test = graphlab.text_analytics.tf_idf(test_examples['word_count'])
test_examples['tfidf'] = tfidf_test['docs']
for i in range(len(test_examples)):

    dictList =[]
    classify = {}
    for j in range(len(train_examples)):

        a = graphlab.distances.cosine(test_examples['tfidf'][i],train_examples['tfidf'][j])
        dictList.append((a,int(train_examples[j]['Prediction'])))
    dictList.sort(key = itemgetter(0))

    top_labels = [x[1] for x in dictList[0:5]]
    classify['author'] = top_labels.count(0)
    classify['movie'] = top_labels.count(1)
    classify['music'] = top_labels.count(2)
    classify['interviews']= top_labels.count(3)
    print classify
    predicted_class = sorted(classify.items(),key = operator.itemgetter(1),reverse=True)
    prediction = predicted_class[0]
    #print prediction
    if prediction[0] == 'author':
        print "The predicted class is author"
        yhat=0
        print yhat
        label = train_examples[j]['Prediction']
        print label
    elif prediction[0] =='movie':
        print "The predicted class is movie"
        yhat = 1
        print yhat
        label = train_examples[j]['Prediction']
        print label

    elif prediction[0] == 'music':
        print "The predicted class is music"
        yhat = 2
        print yhat
        label = train_examples[j]['Prediction']
        print label

    else:
        print "The predicted class is interviews"
        yhat = 3
        print yhat
        label = train_examples[j]['Prediction']
        print label

    if yhat!= label:
        valid_wrong_classif += 1
print "Total errors made are %d"%valid_wrong_classif













