import matplotlib.pyplot as plt
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

##############################################
#   LOAD DATA
###############################################
ortb1 ='we_data/ortb1_xg_bids_validation.csv'
ortb2 ='we_data/ortb2_xg_bids_validation.csv'
linear ='we_data/linear_xg_bids_validation.csv'
linear2 ='we_data/linear2_xg_bids_validation.csv'

ortb1_xg_bids_validation= joblib.load(ortb1)
ortb2_xg_bids_validation = joblib.dump(ortb2)
linear_xg_bids_validation = joblib.dump(linear)
linear2_xg_bids_validation = joblib.dump(linear2)

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
#   WINNING CRITERIA 2 
###############################################
arrays = [ortb1_xg_bids_validation, ortb2_xg_bids_validation, linear_xg_bids_validation,linear2_xg_bids_validation]
a = np.stack(arrays, axis=0)
winning = np.zeros(a.shape)
clicks = np.zeros(a.shape)
second_price = np.zeros(a.shape)

for i in range(len(ortb1_xg_bids_validation)):
  row = a[:,i]
  second_price[:,i] = (row == np.min(heapq.nlargest(2, row))).astype(int)
  winning[:,i] = (row == np.max(row)).astype(int)
  
for i in range(len(ortb1_xg_bids_validation)):
  for j in range(4):
    if winning[j,i] == 1:
      clicks[j,i] += valid_bid_y['click'].values[i]


budget = 6250000
budgetisation = second_price * a
for i in range(len(budgetisation)):
  l = np.zeros(budgetisation[i].shape)
  new = winning[i][np.cumsum(budgetisation[i]) <= budget]
  l[:new.shape[0]] = new
  winning[i] = l
  
wins = np.sum(winning, axis = 1)/1000.
clickwon = np.sum(clicks, axis = 1)
fig, ax = plt.subplots()
modelsname = ["ORTB1","ORTB2","Linear","Linear2"]

ax.barh(np.arange(1,5),wins, label='bids won',
        color='#4682B4')
# Create labels
label = [str(x) + 'clicks' for x in clickwon]
 
# Text on the top of each barplot
for i, v in enumerate(wins):
  msg = str(int(clickwon[i])) + ' clicks'
  ax.text(v , i+1 , msg, color='#CD5C5C', fontweight='bold')

ax.set_yticks(np.arange(1,5))
ax.set_ylim((0.5,4.5))
#ax.set_xlim((0,wins[1]/1000.))
ax.set_yticklabels(modelsname)
#ax.set_xticklabels(wins/1000.)
ax.invert_yaxis()
ax.set_xlabel('winning bids (x1000)')
#ax.set_title('Comparison of bidding strategies')
plt.show()

