import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools
import pandas as pd
import numpy as np
import os
import datetime
from datetime import datetime
from datetime import timedelta
from datetime import date

import scipy 
from scipy import optimize
import scipy.signal as signal 
import sys
import matplotlib.pyplot as plt
    
sys.path.append('../dsmuc/')
from dsmuc.custom import detect_peaks
import dsmuc.io as io
import dsmuc.preprocessing as pp
import dsmuc.features as ff
import dsmuc.custom as cs


import pytz
from azure.storage.blob import BlockBlobService
from io import StringIO
from azure.storage.blob import AppendBlobService
from azure.storage.blob import BlockBlobService
import requests
import json

label_dict = {1:'walking',
             2:'walking upstairs',
             3:'walking downstairs',
             4:'sitting',
             5:'standing',
             6:'laying',
             7:'unknown'}


download_dir =  "../../data/G9_data/Downloaded"
account_name='watchstorage'
account_key='TJWcjsCs4aK9Xorw4DIAZGvKz0AFb2kvgSh49t+3nADR2usZ1ED14GLBQ/klJsSSrKykxu0ghCXn46+0bv2J8Q=='
container_name_ = 'jnj'

# Model saved
filename = './activity-recognition/finalized_model.sav'
logreg = pickle.load(open(filename, 'rb'), encoding = 'iso-8859-1')

def time_to_str(t):
    t_woM = t.replace(microsecond=0)
    dt64 = np.datetime64(t_woM)
    a = dt64.astype('datetime64[s]')
    
    return np.datetime_as_string(a)+"Z"
def f(x):
    u, c = np.unique(x['predictions'].values, return_counts=True)
    outcome = u[np.argmax(c)]
    return outcome


# In[4]:

def do_recognition():

        ##################
    #### READ DATA ###
    ##################
    day_now=0
    day_before=1

    account_name='watchstorage'
    account_key='TJWcjsCs4aK9Xorw4DIAZGvKz0AFb2kvgSh49t+3nADR2usZ1ED14GLBQ/klJsSSrKykxu0ghCXn46+0bv2J8Q=='
    container_name_ = 'jnj'

    blob_service = BlockBlobService(account_name=account_name, account_key = account_key)


    blobs = [];blob_date = []
    generator = blob_service.list_blobs(container_name_)
    for blob in generator:
        blobs.append(blob.name)
        blob_date.append(blob.name[:10])
    blob_table = pd.DataFrame()
    blob_table['date'] = blob_date
    blob_table['blobname'] = blobs    

    today = date.today().strftime('%Y-%m-%d')
    yesterday = (date.today() - timedelta(1)).strftime('%Y-%m-%d')
    blob_table = blob_table[(blob_table['date']==yesterday)|(blob_table['date']==today)] 

    
    if blob_table.shape[0]>0:
        blob_df = pd.DataFrame()
        for blobname in blob_table['blobname']:
            blob_Class = blob_service.get_blob_to_text(container_name=container_name_, blob_name = blobname)
            blob_String =blob_Class.content
            for chunk in pd.read_csv(StringIO(blob_String), chunksize=10000):
                blob_df = blob_df.append(chunk)

        print("READ DATA FRAMES SIZE :",blob_df.shape[0])


    #################
    #################
    feature_list =  ['aoa','ate','apf','rms','std','minimax','cor','mean','min','max']
    preserved_features=['start']


    for watch_id in blob_df['id'].unique()[::-1]:
        print("Watch ", watch_id," is being processed" )
        df_temp = io.read_g9(blob_df[blob_df['id']==watch_id], sort=False)
        df_temp = df_temp.drop_duplicates(keep='last')[::2].sort_index()
        print("READ DATA FRAMES SIZE AFTER CLEANING :",df_temp.shape[0])


        # Time to do analysis is specified
        start = yesterday + 'T16:00:00.0000Z'
        start_temp = np.datetime64(start)
        t = pd.Timestamp(start_temp)
        end = today + 'T16:00:00.0000Z'
        end_temp = np.datetime64(end)
        end_time = pd.Timestamp(end_temp)

        # Initialize 
        whole_window_size = timedelta(minutes = 5)
        window_size = timedelta(seconds=2)
        window_slide = timedelta(seconds=1)
        samples_count = []
        a = 0
        df_out = pd.DataFrame()
        t_start_list = []
        t_end_list = []
        outcome_list = []
        while (t + whole_window_size < end_time):
            label_list = []
            increment = 0
            DF = pd.DataFrame()
            t_end5min= t + whole_window_size 
            print("doing time:",t, ' - ', t_end5min)
            t_start_list.append(time_to_str(t))
            t_end_list.append(time_to_str(t_end5min))
            if df_temp.between_time(t.to_pydatetime().time(), t_end5min.to_pydatetime().time()\
                                               ,include_start=True, include_end=False).shape[0] >= 10:


                while(t+window_slide< t_end5min):
                    t_end = t + window_size
                    snippet_df = df_temp.between_time(t.to_pydatetime().time(), t_end.to_pydatetime().time()
                                                   ,include_start=True, include_end=False)
                    if snippet_df.shape[0]>= 20:
                        increment +=1
                        ser = ff.extract_features(snippet_df, index=increment, feature_list=feature_list ,\
                                    preserved_features=preserved_features)
                        DF = DF.append(ser)
                    t = t_end
            else:
                t = t_end5min

            if DF.shape[0]<=11:
                outcome = 7.0
            else:
                df_X = DF.set_index(pd.DatetimeIndex(DF['start'])).drop('start' ,axis =1)
                del DF 
                df_X.fillna(df_X.mean().fillna(0), inplace=True)
                X_test = df_X.values
                y_pred = logreg.predict(X_test)                
                u, c = np.unique(y_pred, return_counts=True)
                outcome = u[np.argmax(c)]
            outcome_list.append(label_dict[int(outcome)])
            out_ser = pd.Series(outcome,name=(t-whole_window_size, t) )
            df_out = df_out.append(out_ser)
            plt.plot(list(range(df_out.shape[0])), df_out[0], "*")
            ## Send predictions 
        plt.show()   
        dict_list = []
        for i in range(len(outcome_list)):
            payload_dict = {'address':watch_id.split("-")[2],
                 'starttime':t_start_list[i],
                 'endtime':t_end_list[i],
                 'tasklocation':'Activity',
                 'taskname':outcome_list[i],
                 'name':outcome_list[i],
                 'value':1}
            dict_list.append(payload_dict)
        payload = json.dumps(dict_list)
        url = "https://colife-dashboard.silverline.mobi/uploadActivityLabelForSmartWatch"
        headers = {
            'content-type': "application/json",
            'cache-control': "no-cache",
            'postman-token': "87b2b04f-175f-4a9b-f2c8-bf31de2cae7d"
            }

        response = requests.request("POST", url, data=payload, headers=headers)
        print(response.text)


    return True

