import os
import pandas as pd
from math import sqrt
import mlflow
from scipy import fftpack
import librosa
import matplotlib.pyplot as plt
import numpy as np
import math
import statistics

mlflow.set_tracking_uri("http://localhost:5000/")


# ML experiment name
mlflow.set_experiment("AcouWood")

pd.set_option('display.max_rows', None)

df_features = []
classes = []

with mlflow.start_run(run_name="KNN w. Standardscaler"): 

    def features(df1):
        #should calculate features and therefore take df as parameter

        df_freq = df1['Frequency (Hz)']
        df_amp = df1['Level (dB)']
        mean_all = df_amp.mean()
        mx_all = df_amp.max()
        mn_all = df_amp.min()
        
        df_amp_5 = df_amp[df_freq < 5000] # Ampltitude data only below 5 khz
        df_amp_5_10 = df_amp[(5000 <= df_freq) & (df_freq < 10000)] 
        df_amp_10_15 = df_amp[(10000 <= df_freq) & (df_freq < 15000)]
        df_amp_15_20 = df_amp[(15000 <= df_freq) & (df_freq < 20000)]
        df_amp_20 = df_amp[20000 <= df_freq]

        df_list = [df_amp_5,df_amp_5_10,df_amp_10_15,df_amp_15_20,df_amp_20]
    
        features = []
        for df in df_list:
            mean = df.mean()
            mx = df.max()
            mn = df.min()
            features.extend((mean,mx,mn))
        df_features.append(features)
        return df_features

    def qmean(num):
        return sqrt(sum(n*n for n in num)/len(num))

    def spectral(y, sr):
        sp = librosa.feature.spectral_centroid(y=y, sr=sr)[0] # Compute all centroid frequencies and store in array
        sp_length = len(sp) 
        sp_mean = np.mean(sp)
        sp_median = np.median(sp)

        sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0] # Compute all centroid frequencies and store in array
        sb_length = len(sb) 
        sb_mean = np.mean(sb)
        sb_median = np.median(sb)

        features = [sp_length,sp_mean,sp_median,sb_length,sb_mean,sb_median]
        data.append(features)

    def FFT(data,sr):
        #Plot the Signal
        N = 2048
        T = 1.0/sr
        value = np.fft.rfft(data, N)
        MAG = np.abs(value)# Apply the fft function  
        MAG_dB = np.abs(20*np.log10(MAG/max(value)))
        freq = np.fft.rfftfreq(N, d=T) 
        df1 = pd.DataFrame(freq,columns=["Frequency (Hz)"])
        df1['Level (dB)'] = MAG_dB
        #print(df1)
        #plt.plot(freq, MAG_dB)
        #plt.show()
        return df1

    # The directory of audio files
    dir = 'wav/'

    for filename in os.listdir(dir):
        c = filename[4:5]
        if c == "d" or c == "e":
            continue
        else:
            filename_path = os.path.join(dir, filename)
            data, sr = librosa.load(filename_path, sr=44100)
            df1 = FFT(data,sr)
            data = features(df1)
            classes.append(c)

    # 'sp_length','sp_mean','sp_median','sb_length','sb_mean','sb_median',
    columns_= ['mean_5', 'max_5', 'min_5', 'mean_5_10', 'max_5_10', 
            'min_5_10', 'mean_10_15', 'max_10_15', 'min_10_15', 
            'mean_15_20', 'max_15_20','min_15_20','mean_20','max_20','min_20']

    df2 = pd.DataFrame(data, columns=columns_) # Store calculated features with feature names in dataframe
    df2['class'] = classes # Adding column with class label

    
    # Check the proportion of "n" and "m" observations in the dependent variable target
    # print(df2['class'].value_counts()/len(df2['class']))   
    # plt.hist(df2['mean_5'], bins=20)
    # plt.show()

    #####################################################################################3
    # Import sklearn modules
    from sklearn.pipeline import Pipeline
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
    from sklearn.metrics import classification_report


    # Define the name of features used
    feature_names = columns_
    # Define name of target variable
    class_names = ['class']


    # Encode categorical target variable
    le = LabelEncoder()
    df2['class'] = le.fit_transform(df2['class'])

    # Store data from df in X and y variables 
    X = df2[feature_names]
    y = df2[class_names]

    class Debug(BaseEstimator, TransformerMixin):
        def fit(self,X,y): return self
        def transform(self,X): 
            print(X)
            return X

    pipeline = Pipeline([
                      ("columns", ColumnTransformer([
                          ("scaler", StandardScaler(), columns_),
                          ], remainder="passthrough")),
                    #("encoder", OneHotEncoder()),
                    #("debug", Debug()),   
                    #("PCA", PCA(n_components=8)),
                    ("model", KNeighborsClassifier()),     
                ])

    number_of_splits = 5

    metrics = [("Accuracy", accuracy_score, []), ("Precision", precision_score, []), ("Recall", recall_score, [])]

    mlflow.log_param("number_of_splits", number_of_splits)
    mlflow.log_param("transformers", pipeline["columns"])
    mlflow.log_param("model", pipeline["model"])

    # Cross validation 
    for train, test in StratifiedKFold(number_of_splits).split(X,y):
        pipeline.fit(X.iloc[train], y.iloc[train].values.ravel())
        predictions = pipeline.predict(X.iloc[test])
        truth = y.iloc[test]
                    
        # Calculate and save the metrics for this fold
        for name, func, scores in metrics:
            score = func(truth, predictions)
            scores.append(score)

    # Log summary of the metrics
    for name, _, scores in metrics:
        mean_score = sum(scores)/number_of_splits # The mean of the scores
        std_score = statistics.stdev(scores) # The standard deviation of the scores
        mlflow.log_metric(f"mean_{name}", mean_score)
        mlflow.log_metric(f"std_{name}", std_score)  
        print(f"mean_{name}", mean_score)  

    #print(pipeline['PCA'].components_)