'''
COMP-598 - Applied Machine Learning 
Project 2 - Classification

defines.py = Contains global definitions
@authors: Sandra Maria Nawar
          Timardeep Kaur
          Daniel Galeano
October 20th, 2015
'''

#*******************************************************************************
#*******************************************************************************
import numpy as np

# DEFINITIONS
#DATA_TRAIN_CSV_FILE = "train.csv"
#DATA_TEST_CSV_FILE = "test.csv"
#DATA_TRAIN_CSV_FILE = "ml_dataset_train.csv"
#DATA_TEST_CSV_FILE = "ml_dataset_test_in.csv"
DATA_TRAIN_CSV_FILE = "train.csv"
DATA_TEST_CSV_FILE = "test.csv"

LABEL_AUTHOR    = 0
LABEL_MOVIES    = 1
LABEL_MUSIC     = 2
LABEL_INTERVIEW = 3

FEATURE_SELECTED = 500


SELECTED_FEATURES = [
'book',
'say',
'write',
'into',
'read',
'author',
'story',
'novel',
'years',
'talk',
'actually',
'life',
'sort',
'books',
'world',
'back',
'thanks',
'call',
'writing',
'said',
'thank',
'fact',
'point',
'war',
'called',
'question',
'interesting',
'stories',
'reading',
'course',
'talking',
'thought',
'film',
'movie',
'character',
'scene',
'films',
'actor',
'director',
'movies',
'directed',
'documentary',
'actors',
'see',
'funny',
'script',
'acting',
'role',
'comedy',
'scenes',
'characters',
'clip',
'plays',
'oscar',
'watch',
'shot',
'camera',
'starring',
'stars',
'filmmaker',
'show',
'shoot',
'watching',
'music',
'song',
'album',
'songs',
'band',
'record',
'play',
'hear',
'singing',
'playing',
'sound',
'sing',
'love',
'singer',
'cd',
'guitar',
'recording',
'jazz',
'listen',
'musicians',
'rock',
'musical',
'recorded',
'records',
'lyrics',
'musician',
'piano',
'blues',
'studio',
'heard',
'played',
'voice',
'president',
'government',
'time',
'obama',
'look',
'american',
'country',
'issue',
'day',
'congress',
'states',
'money',
'administration',
'whether',
'united',
'state',
'support',
'white',
'michel',
'issues',
'national',
'times',
'senator',
'republican',
'public',
'help',
'law',
'percent',
'policy',
'house',
'iraq',
'community'
]


LABELS = np.array([LABEL_AUTHOR, LABEL_MOVIES, 
                   LABEL_MUSIC, LABEL_INTERVIEW])

