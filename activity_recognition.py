
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import datetime
import scipy 
from scipy import optimize
import scipy.signal as signal 
from detect_peaks import *
import quaternion as quat


from azure.storage.blob import BlockBlobService
import requests
import json
download_dir =  "G9_data/Downloaded"
label_dict = {1:'walking',
             2:'walking upstairs',
             3:'walking downstairs',
             4:'sitting',
             5:'standing',
             6:'laying',
             7:'unknown'}


# In[2]:


# Trained Model
df_model = pd.read_csv("modelbuilding/data/preprocessed/data.csv", index_col=0)
df_model.dropna(axis=0, how='any', inplace=True)



# In[4]:

def do_recognition():

    X_training = df_model[df_model.columns[:-2]].values
    y_training = df_model['label'].values


    # In[5]:


    logreg = LogisticRegression(penalty = 'l1')
    logreg.fit(X_training, y_training)


    # In[6]:


    interested_cols = [ 'AccX', 'AccY', 'AccZ', 'GyroX','GyroY', 'GyroZ']
    def create_datetime_index(df):
        # convert unix time to german time and then to date
        df['date'] = pd.to_datetime(df['system_time'],unit='ms')
        # set datetime index
        df.index = pd.DatetimeIndex(df["date"])
        df = df.drop(labels=["date"],  axis=1)
        return df
    def read_file(path):
        df = pd.read_csv(path)
        return df
    def get_sensor_data(df):

        df = create_datetime_index(df)
        df.sort_index(ascending = True, inplace = True)

        # Sensor Selection
        df_acc = df[df['sensor_name']=='Accelerometer']
        df_gyro = df[df['sensor_name']=='Gyroscope']

        # Merge accelerometer and gyroscope dataframes
        df_merged = pd.merge(df_acc, df_gyro, left_index=True, right_index= True, how='inner')
        df_merged.drop([
         'sensor_name_x',
         'value_x',
         'id_y',
         'sensor_name_y',
         'system_time_y',
         'value_y'], axis = 1, inplace = True)

        # Drop duplicates
        df_merged = df_merged.loc[::2] # run ony once

        # new column names
        df_merged.columns = ['id','system_time', 'AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']
        df_merged.sort_index(ascending = True, inplace=True)
        return df_merged
    def find_nearest(array,value):
        idx = (np.abs(array-value)).argmin()
        return idx
    def extract_energy_variables(f, psdX):

        i_low = find_nearest(f,4.0)
        i_high = find_nearest(f,7.0)

        energy_total = np.sum(psdX) 
        energy_interested = np.sum(psdX[i_low : i_high + 1]) 
        max_total = np.max(psdX)
        max_interested = np.max(psdX[i_low : i_high + 1])
        return energy_total,energy_interested, max_total, max_interested
    def average_over_axis(df):
        aoa = df[interested_cols].mean(axis = 0)
        aoa.index += '_aoa'
        return aoa
    def average_time_elapse(df):
        list_= []
        for col in interested_cols:
            a = df[col].values
            mph = a.mean()
            ind = detect_peaks(a, mph = mph, mpd=10, show=False)
            list_.append(np.diff(ind).mean())
        ate = pd.Series(list_, index=interested_cols)
        ate.index += '_ate'
        ate[pd.isnull(ate)]=0 #### TODO
        return ate
    def average_peak_freq(df):
        list_f= []
        for col in interested_cols:
            a = df[col].values
            mph = a.mean()
            ind = detect_peaks(a, mph = mph, mpd=10, show=False)
            list_f.append(len(ind)/a.shape[0])
        apf = pd.Series(list_f, index=interested_cols)
        apf.index += '_apf'
        return apf
    def rms_func(df):
        list_= []
        for col in interested_cols:
            a = df[col].values
            rms_temp = np.sqrt(np.mean(a**2))
            list_.append(rms_temp)
        rms = pd.Series(list_, index=interested_cols)
        rms.index += '_rms'
        return rms
    def std_func(df):
        list_= []
        for col in interested_cols:
            a = df[col].values
            std_temp = np.std(a)
            list_.append(std_temp)
        std = pd.Series(list_, index=interested_cols)
        std.index += '_std'
        return std
    def minmax_func(df):
        list_= []
        for col in interested_cols:
            a = df[col].values
            minmax_temp = np.max(a)-np.min(a)
            list_.append(minmax_temp)
        minmax = pd.Series(list_, index=interested_cols)
        minmax.index += '_minmax'
        return minmax
    def cor_func(df):
        a = df[interested_cols[:3]].corr()
        b= df[interested_cols[3:]].corr()
        indexes = ['CorAccXAccY','CorAccXAccZ','CorAccYAccZ', 'CorGyroXGyroY','CorGyroXGyroZ','CorGyroYGyroZ']
        Cor = (a['AccX'][1:]).append(a['AccY'][2:]).append((b['GyroX'][1:]).append(b['GyroY'][2:]))
        Cor[pd.isnull(Cor)]=0 ### TODO
        corr = pd.Series(Cor.values, indexes)
        corr.index += '_corr'
        return corr
    def get_all_features(df, file):

        aoa = average_over_axis(df)
        ate = average_time_elapse(df)
        apf = average_peak_freq(df)
        rms = rms_func(df)
        std = std_func(df)
        minmax = minmax_func(df)
        cor = cor_func(df)
        ser_list = [aoa, ate,apf, rms,std, minmax, cor]
        ser = pd.concat(ser_list)
        ser.name = file
        return ser




    # In[7]:


    def time_to_str(t):
        t_woM = t.replace(microsecond=0)

        dt64 = np.datetime64(t_woM)
        dt64

        a = dt64.astype('datetime64[s]')

        return np.datetime_as_string(a)+"Z"


    # In[9]:


    block_blob_service = BlockBlobService(account_name='watchstorage', account_key='TJWcjsCs4aK9Xorw4DIAZGvKz0AFb2kvgSh49t+3nADR2usZ1ED14GLBQ/klJsSSrKykxu0ghCXn46+0bv2J8Q==')

    container_name = 'jnj'
    generator = block_blob_service.list_blobs(container_name)

    # Download each day movement data 
    for blob in generator:
        print("\t Blob name: " + blob.name)
        if not os.path.exists(os.path.join(download_dir, blob.name)) : # check filecmp.cmp()
            if not os.path.exists(os.path.join(download_dir, blob.name.split('/')[0])):
                os.mkdir(os.path.join(download_dir, blob.name.split('/')[0]))
            print("downloading...")
            block_blob_service.get_blob_to_path(container_name, blob.name,os.path.join(download_dir, blob.name ))
            print("downloaded")
        else:
            print("Already downloaded ")


    # In[10]:


    from datetime import date, timedelta
    yesterday = date.today() - timedelta(1)
    yesterday = yesterday.strftime('%Y-%m-%d')
    today = date.today().strftime('%Y-%m-%d')
    dates_to_consider = [(date.today() - timedelta(i)).strftime('%Y-%m-%d') for i in range(0,6)]
    print(yesterday)
    print(today)


    # In[11]:


    df_list = []
    for day in dates_to_consider:
        path = os.path.join(download_dir, day)
        if os.path.exists(path):
            for root,dirs,files in os.walk(path):
                if files[0].endswith(".csv"):
                    file_path = path+'/'+files[0]
                    df_list.append(pd.read_csv(file_path))
    df = pd.concat(df_list)


    # In[12]:


    df.head()


    # In[13]:


    df['id'].unique()


    # In[ ]:


    for watch_id in df['id'].unique():
        #TODO: Select all watches and do the predictions
        print("Watch ", watch_id," is being processed" )
        df_temp = get_sensor_data(df)
        # TODO: Select Sensor name
        df_temp = df_temp[df_temp['id']==watch_id]
        df_temp.sort_index(ascending = True, inplace = True)
        # Time to do analysis is specified
        start = yesterday + 'T16:00:00.0000Z'
        start_temp = np.datetime64(start)
        t = pd.Timestamp(start_temp)
        end = today + 'T16:00:00.0000Z'
        end_temp = np.datetime64(end)
        end_time = pd.Timestamp(end_temp)

        # Initialize 
        whole_window_size = datetime.timedelta(minutes = 5)
        window_size = datetime.timedelta(seconds=2)
        window_slide = datetime.timedelta(seconds=1)
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
            while(t+window_slide< t_end5min):
                t_end = t + window_size
                snippet_df = df_temp.between_time(t.to_pydatetime().time(), t_end.to_pydatetime().time()
                                               ,include_start=True, include_end=False)
                if snippet_df.shape[0]>= 20:
                    increment +=1
                    ser = get_all_features(snippet_df, increment)
                    ser = ser.round(4)
                    DF = DF.append(ser, verify_integrity=True)
                t = t_end
            #DF.dropna(axis=0, how='any', inplace=True)
            DF = DF.fillna(DF.mean())
            if DF.shape[0]<=11:
                outcome = 7.0
            else:
                X_test = DF.values
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

