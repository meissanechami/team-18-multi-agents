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
#   BIDDING FUNCTIONS
###############################################
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
#    GET CTRs
###############################################
model = xgboost.XGBClassifier(n_estimators = 500, max_depth = 8, learning_rate=0.1)
model.fit(train_bid_x,train_bid_y["click"])

#clfname = 'Tuned_XGB_newprepro.csv'
#joblib.dump(model, clfname, compress=9)
#model = joblib.load(clfname)

##############################################
#    APPLY BIDDING STRATEGIES
###############################################
pCTR_valid_xg = model.predict_proba(valid_bid_x)
v = model.predict(valid_bid_x)
avgctr= sum(v)/len(v)

ortb1_xg_bids_validation = []
ortb2_xg_bids_validation = []
linear_xg_bids_validation = []
linear2_xg_bids_validation = []
count = 0
cvg = np.asarray(pCTR_valid_xg)
avgc = sum(v)/len(v)

for i in range(0,len(cvg)):
    click_probab = cvg[i,1]
    bid1 = predict_optimal_bid1(click_probab,2.)
    bid2 = predict_optimal_bid2(click_probab,8.)
    bid3 = (click_probab/avgc) * 23.389
    bid4 = bidding_linear1(click_probab, avgc, 78.)
    ortb1_xg_bids_validation.append(bid1)
    ortb2_xg_bids_validation.append(bid2)
    linear_xg_bids_validation.append(bid3)
    linear2_xg_bids_validation.append(bid4)


ortb1 ='we_data/ortb1_xg_bids_validation.csv'
ortb2 ='we_data/ortb2_xg_bids_validation.csv'
linear ='we_data/linear_xg_bids_validation.csv'
linear2 ='we_data/linear2_xg_bids_validation.csv'

joblib.dump(ortb1_xg_bids_validation, ortb1, compress=9)
joblib.dump(ortb2_xg_bids_validation, ortb2, compress=9)
joblib.dump(linear_xg_bids_validation,linear, compress=9)
joblib.dump(linear2_xg_bids_validation,linear2, compress=9)

##############################################
#    COMPARE BIDDING
###############################################

# Winning criteria 1
budget = 0
wins_1 = 0
cc = 0 
clicks_1 = 0
for i in range(len(ortb1_xg_bids_validation)):
  if ortb1_xg_bids_validation[i] > valid_bid_y['payprice'].values[i]:
    wins_1 +=1
    if valid_bid_y['click'].values[i] == 1:
      clicks_1 += valid_bid_y['click'].values[i]
      cc += valid_bid_y['payprice'].values[i]
    budget += valid_bid_y['payprice'].values[i]
  if budget >= 6250000:
    break

CTR = clicks_1/float(len(ortb1_xg_bids_validation))
CPM = budget/float(len(ortb1_xg_bids_validation))
CPC = cc/clicks_1
spent = budget
print("ORTB1: wining bids: %f  clicks: %f  spent: %f  avgCTR: %f  avgCPM: %f  avgCPC: %f"%(wins_1,clicks_1,spent, CTR, CPM, CPC))

# Winning criteria 1
budget = 0
wins_1 = 0
cc = 0
clicks_1 = 0
for i in range(len(ortb2_xg_bids_validation)):
  if ortb2_xg_bids_validation[i] > valid_bid_y['payprice'].values[i]:
    wins_1 +=1
    if valid_bid_y['click'].values[i] == 1:
      clicks_1 += valid_bid_y['click'].values[i]
      cc += valid_bid_y['payprice'].values[i]
    budget += valid_bid_y['payprice'].values[i]
  if budget >= 6250000:
    break

CTR = clicks_1/float(len(ortb2_xg_bids_validation))
CPM = budget/float(len(ortb2_xg_bids_validation))
CPC = cc/clicks_1
spent = budget
print("ORTB2: wining bids: %f  clicks: %f  spent: %f  avgCTR: %f  avgCPM: %f  avgCPC: %f"%(wins_1,clicks_1,spent, CTR, CPM, CPC))

# Winning criteria 1
budget = 0
wins_1 = 0
cc = 0
clicks_1 = 0
for i in range(len(linear_xg_bids_validation)):
  if linear_xg_bids_validation[i] > valid_bid_y['payprice'].values[i]:
    wins_1 +=1
    if valid_bid_y['click'].values[i] == 1:
      clicks_1 += valid_bid_y['click'].values[i]
      cc += valid_bid_y['payprice'].values[i]
    budget += valid_bid_y['payprice'].values[i]
  if budget >= 6250000:
    break

CTR = clicks_1/float(len(linear_xg_bids_validation))
CPM = budget/float(len(linear_xg_bids_validation))
CPC = cc/clicks_1
spent = budget
print("LINEAR: wining bids: %f  clicks: %f  spent: %f  avgCTR: %f  avgCPM: %f  avgCPC: %f"%(wins_1,clicks_1,spent, CTR, CPM, CPC))

# Winning criteria 1
budget = 0
wins_1 = 0
cc = 0
clicks_1 = 0
for i in range(len(linear2_xg_bids_validation)):
  if linear2_xg_bids_validation[i] > valid_bid_y['payprice'].values[i]:
    wins_1 +=1
    if valid_bid_y['click'].values[i] == 1:
      clicks_1 += valid_bid_y['click'].values[i]
      cc += valid_bid_y['payprice'].values[i]
    budget += valid_bid_y['payprice'].values[i]
  if budget >= 6250000:
    break

CTR = clicks_1/float(len(linear2_xg_bids_validation))
CPM = budget/float(len(linear2_xg_bids_validation))
CPC = cc/clicks_1
spent = budget
print("LINEAR2: wining bids: %f  clicks: %f  spent: %f  avgCTR: %f  avgCPM: %f  avgCPC: %f"%(wins_1,clicks_1,spent, CTR, CPM, CPC))


df = pd.DataFrame()
df["ORTB1"] = ortb1_xg_bids_validation
df["ORTB2"] = ortb2_xg_bids_validation
df["LINEAR"] =linear_xg_bids_validation
df["LINEAR2"] =linear2_xg_bids_validation
df["click"] = valid_bid_y["click"]

filename_theta = 'we_data/ensemble_bids.csv'
df.to_csv(filename_theta, index=False)




