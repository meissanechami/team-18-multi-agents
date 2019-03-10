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
#   LOGISTIC REGRESSION
###############################################
import sklearn
from sklearn.linear_model import LogisticRegression

clf = sklearn.linear_model.LogisticRegression(C = 0.1).fit(train_bid_x, train_bid_y["click"])
# clfname = '/content/drive/My Drive/Multi-Agent/Tuned_clf.csv'
# joblib.dump(clf, clfname, compress=9)
print(metrics.balanced_accuracy_score(train_bid_y['click'],  clf.predict(train_bid_x)))
print(metrics.balanced_accuracy_score(valid_bid_y['click'], clf.predict(valid_bid_x)))

##############################################
#   Extra Trees Classifier
###############################################
from sklearn.ensemble import  ExtraTreesClassifier
modeletc = ExtraTreesClassifier(n_estimators=500, criterion='gini', max_depth=30, min_samples_split=2,\
                                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', \
                                 max_leaf_nodes=None, min_impurity_split=1e-07, \
                                 bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, \
                                 warm_start=False, class_weight=None)
modeletc.fit(train_bid_x,train_bid_y["click"])

print(metrics.balanced_accuracy_score(train_bid_y['click'],  modeletc.predict(train_bid_x)))
print(metrics.balanced_accuracy_score(valid_bid_y['click'], modeletc.predict(valid_bid_x)))
##############################################
#   XGBoost
###############################################
model = xgboost.XGBClassifier(n_estimators = 500, max_depth = 8, learning_rate=0.1)
model.fit(train_bid_x,train_bid_y["click"])

print(metrics.balanced_accuracy_score(train_bid_y['click'],  model.predict(train_bid_x)))
print(metrics.balanced_accuracy_score(valid_bid_y['click'], model.predict(valid_bid_x)))
##############################################
#   LSTM
###############################################
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense,Dropout, Masking, Embedding, Input, RNN, Flatten


fulldata = pd.concat([train_bid_x,valid_bid_x])
fulllabel = pd.concat([train_bid_y[["click"]],valid_bid_y[["click"]]])
c_train= fulldata.as_matrix() 
c_test= test_bid_x.as_matrix() 


X_train = np.reshape(c_train, (c_train.shape[0], c_train.shape[1], 1))
X_test = np.reshape(c_test, (c_test.shape[0], c_test.shape[1], 1))

lstm = Sequential()
lstm.add(LSTM(128, return_sequences=True, input_shape = (X_train.shape[1], 1)))
lstm.add(LSTM(64, return_sequences=True))
lstm.add(Dropout(0.2))
lstm.add(Flatten())
lstm.add(Dense(1, activation="sigmoid"))

lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm.fit(X_train, fulllabel.values, epochs=2, batch_size=126)

print(metrics.balanced_accuracy_score(train_bid_y['click'],  lstm.predict(train_bid_x)))
print(metrics.balanced_accuracy_score(valid_bid_y['click'], lstm.predict(valid_bid_x)))


