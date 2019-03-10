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

#pull data
filename_train = 'we_data/train.csv'
filename_valid = 'we_data/validation.csv'
filename_test = 'we_data/test.csv'

df = pd.read_csv(filename_train)
df_valid = pd.read_csv(filename_valid)
df_test = pd.read_csv(filename_test)
#random sampling
#df = df.sample(frac=1)
#df_valid =df_valid.sample(frac=1)
#df_test = df_test.sample(frac=1)

# Random downsampling of the dataset
train_dummy_outliers = df.loc[df["click"] == 1]
train_dummy_inliers = df.loc[df["click"] == 0].sample(n = 40000, random_state = 123).reset_index(drop = True)
    
train_dummy_2 = pd.concat([train_dummy_inliers, train_dummy_outliers], axis = 0)
df_train_x = train_dummy_2.loc[:, ((train_dummy_2.columns != "click") & 
                      (train_dummy_2.columns != "payprice") &
                      (train_dummy_2.columns != "bidprice"))]
df_train_y = train_dummy_2[['click','payprice','bidprice']]

df_valid_x = df_valid.loc[:, ((df_valid.columns != "click") & 
                                        (df_valid.columns != "payprice") & 
                                        (df_valid.columns != "bidprice"))]
df_valid_y = df_valid[['click','payprice','bidprice']]


df_test_x = df_test

def preprocess_ipinyou(df):
    print( df["useragent"].head())

    df["os"] = df["useragent"].map(lambda x: x.split("_")[0])
    df["browser"] = df["useragent"].map(lambda x: x.split("_")[1])
    df["slotarea"] = df["slotwidth"]*df["slotheight"]   
    print('slot done')
    
    #usertag_preprocessing
    df["usertag"] = df["usertag"].fillna("")  
    df_u = df["usertag"].dropna().reset_index(drop = True)
    usertags_list = [df_u[i].split(",") for i in range(df_u.shape[0])]
    usertags = np.unique(list(itertools.chain.from_iterable(usertags_list)))
    usertags = [tag for tag in usertags if len(tag) > 0]
    for tag in usertags:
        col_name = "usertag_" + tag
        df[col_name] = df["usertag"].map(lambda x: 1 if tag in x.split(",") else 0)
    print('usertag done')
    
    #fill missing values
    df["adexchange"] = df["adexchange"].fillna(df["adexchange"].dropna().mode()[0])
    df[['url','domain','keypage']] = df[['url','domain','keypage']].fillna(value=-1)
    df['slotformat'] = pd.to_numeric(df['slotformat'], errors='coerce').interpolate()
    print('missing done')
    
    # categorising
    id_col = ['adexchange','advertiser','slotvisibility','browser','os','creative']
    le = preprocessing.LabelEncoder()
    for i in id_col:
        df[i] = le.fit_transform(df[i].astype(str))
    print("cat dome")
    
    # one hot encoding
    #strings_col = []
    #for i in strings_col:
    #    for j in df[i].unique():
    #        col_name = i + '_'+ str(j)
    #        df[col_name] = np.where(df[i]==j, 1, 0)
    #print('strings_col done')
    
    # Slotprice binning
    df["slotprice_cat"] = 0

    df.loc[ df["slotprice"] <= 10, "slotprice_cat"] = 0
    df.loc[ (df["slotprice"] > 10) & (df["slotprice"] <= 50), "slotprice_cat"] = 1
    df.loc[ (df["slotprice"] > 50) & (df["slotprice"] <= 100), "slotprice_cat"] = 2
    df.loc[ df["slotprice"] > 100, "slotprice_cat"] = 3

    df = df.drop(columns=['usertag', 'urlid',"slotprice",'useragent','keypage',
                         "slotwidth",'slotheight','bidid', 'userid', 'url','domain','IP','slotid'])
    return df

df_cc = pd.concat([df_train_x, df_valid_x, df_test_x])
shape_train = df_train_x.shape[0]
shape_valid = df_valid_x.shape[0]
shape_test = df_test_x.shape[0]
cc = preprocess_ipinyou(df_cc)

valid_bid_x = cc[shape_train:shape_train+shape_valid]
valid_bid_y = df_valid_y

train_bid_x = cc[:shape_train]
train_bid_y = df_train_y

test_bid_x = cc[shape_train+shape_valid:shape_test+shape_train+shape_valid]

filetrain ='we_data/train_processed.csv'
filevalid ='we_data/valid_processed.csv'
filetest ='we_data/test_processed.csv'
filetrainy ='we_data/train_processed_y.csv'
filevalidy ='we_data/valid_processed_y.csv'

joblib.dump(train_bid_x, filetrain, compress=9)
joblib.dump(valid_bid_x, filetrain, compress=9)
joblib.dump(test_bid_x, filetrain, compress=9)
joblib.dump(train_bid_y,filetrainy, compress=9)
joblib.dump(valid_bid_y,filevalidy, compress=9)













