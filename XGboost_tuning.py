import numpy as np
import pandas as pd
from google.colab import files, drive
import heapq
from imblearn.under_sampling import TomekLinks, RandomUnderSampler, NearMiss
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.externals import joblib
from sklearn import preprocessing
import itertools
import math
from sklearn import metrics


##############################################
#   LOAD DATA
###############################################
filetrain ='we_data/train_processed.csv'
filevalid ='we_data/valid_processed.csv'
filetest ='we_data/test_processed.csv'
filetrainy ='we_data/train_processed_y.csv'
filevalidy ='we_data/valid_processed_y.csv'

train_bid_x = joblib.load(filetrain)
valid_bid_x = joblib.load(filevalid)
test_bid_x= joblib.load( filetest)
train_bid_y = joblib.load(filetrainy)
valid_bid_y = joblib.load(filevalidy)



##############################################
#   RANDOM SEARCH
###############################################

results = []
for i in range(50):
    n_estimator = np.random.randint(300,600)
    max_depth_ =  np.random.randint(3,16)
    learning_rate = np.random.uniform(0.001, 0.1)
    model = xgboost.XGBClassifier(n_estimators=n_estimator, max_depth=max_depth_,learning_rate=learning_rate)
    model.fit(train_bid_x, train_bid_y["click"])
    pred = model.predict(valid_bid_x)
    accuracy = metrics.balanced_accuracy_score(valid_bid_y['click'],pred)
    result = [i,accuracy, n_estimator,max_depth_,learning_rate]
    results.append(result)
    print(result)

