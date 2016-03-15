from sklearn.svm import SVC
from decimal import *
import defines
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def run_crossval(TRAIN_DATA):

  print('get counters!')
  [x_train, y_train] = get_counters(TRAIN_DATA)

  k=4
  x_train=np.matrix(x_train)
  y_train=np.matrix(y_train)
  num_abstracts=len(x_train)
  k_size = num_abstracts/k
  test_error=[]
  train_error=[]
  for i in range (1,k+1):
      if (i==1):
         train_abstracts=x_train[k_size:]
         test_abstracts=x_train[:k_size-1]
         train_output=y_train[k_size:]
         test_output=y_train[:k_size-1]
      elif(i==k):
         train_abstracts=x_train[:(k-1)*k_size]
         test_abstracts=x_train[(k-1)*k_size+1:]
         train_output=y_train[:(k-1)*k_size]
         test_output=y_train[(k-1)*k_size+1:]
      else:
         train_abstracts=x_train[:(i-1)*k_size]+x_train[i*k_size+1:]
         test_abstracts=x_train[(i-1)*k_size+1:i*k_size]
         train_output=y_train[:(i-1)*k_size]+y_train[i*k_size+1:]
         test_output=y_train[(i-1)*k_size+1:i*k_size]
      predicted_test, predicted_train=SVMClassifier(train_abstracts,train_output,test_abstracts)
      test_error.append(SVMerror(predicted_test, test_output))
      train_error.append(SVMerror(predicted_train,train_output))

      f = open('SVMresultsOptimal.txt','a')
      f.write('Test error is: %r: \n' % test_error[i-1])
      f.write('Train error is: %r: \n\n\n' % train_error[i-1])
      f.close()
      print(test_error)

      print(i)


def SVMClassifier(train_abstracts, train_output, test_abstracts):

    clf = SVC( kernel='linear')

    print train_abstracts.shape
    print train_output.shape

    clf.fit(train_abstracts, train_output)
    predicted_train = []
    predicted_test  = []
    
    for sample_train,sample_test in zip(train_abstracts,test_abstracts):
        predicted_train.append(clf.predict([sample_train]))
        predicted_test.append(clf.predict([sample_test]))
        
    return predicted_test, predicted_train
    
def SVMerror(predicted_output,test_output):
    sum=0;    
    for (a,b) in zip(predicted_output,test_output):
        if (a==b) :
            sum +=1
    error = Decimal(sum)/len(predicted_output)
    return error



def get_counters(TRAIN_DATA):
    samples = [ row[0] for row in TRAIN_DATA ]
    labels_train  = [ row[1] for row in TRAIN_DATA ]
    labels_train = [int(i) for i in labels_train]
    labels_train = np.asarray(labels_train)


    #Vectorize and encode selected features/words
    data_vect = CountVectorizer(vocabulary = defines.SELECTED_FEATURES,
                                 min_df = 1,
                                 strip_accents = 'ascii',
                                 analyzer = 'word',
                                 stop_words = 'english' )

    data_fit = data_vect.fit_transform(samples)

    # counters for each sample
    all_c_train = data_fit.toarray()

    print('done preparing data')

    return (all_c_train, labels_train)