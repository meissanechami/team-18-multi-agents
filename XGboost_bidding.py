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

def predict_optimal_bid1(click_probab, c=3.):
    #c = 3.
    a = 5.2*(10**(-7))
    bid = math.sqrt((c/a)*click_probab + c**2) - c
    return bid
  
def predict_optimal_bid2(click_probab ,c=19.):
    #c = 19.
    a = 5.2*(10**(-7))
    term1 = (click_probab+ math.sqrt(c**2 * a**2 + click_probab**2))/(c * a)
    term2 = (c * a)/(click_probab+ math.sqrt(c**2 * a**2 + click_probab**2))
    return c * (term1**(1./3) - term2**(1./3))
  

def bidding_linear1(pred_ctrs, ctrs, cpms):
    if pred_ctrs / ctrs >= 1.2:
        bid = 2 * cpms
    elif 0.8 < float(pred_ctrs / ctrs) < 1.2:
        bid = cpms
    else:
        bid = 0
    return bid

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

import xgboost

model = xgboost.XGBClassifier(n_estimators = 500, max_depth = 8, learning_rate=0.1)
model.fit(train_bid_x,train_bid_y["click"])

#clfname = 'Tuned_XGB_newprepro.csv'
#joblib.dump(model, clfname, compress=9)
#model = joblib.load(clfname)

pCTR_valid_xg = model.predict_proba(valid_bid_x)
v = model.predict(valid_bid_x)
avgctr= sum(v)/len(v)