import os
import pandas as pd
from math import sqrt
import mlflow
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy import fftpack
import librosa

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import r2_score


# ML experiment name
#mlflow.set_experiment("AcouWood")

pd.set_option('display.max_rows', None)

dir = 'wav/'

classdict = {
  "n": 0,
  "m": 1,
  "d": 2,
  "e": 2
}

data = []
classes = []

#with mlflow.start_run(run_name="first_run"): 

def above18(df1):
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
    data.append(features)
    return data

def qmean(num):
    return sqrt(sum(n*n for n in num)/len(num))

def spectral_centroid(y, sr):
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

for filename in os.listdir(dir):
    cl = classdict[filename[4:5]]
    if cl == 2:
        continue
    else:
        filename_path = os.path.join(dir, filename)
        #data = above18(df1)
        classes.append(cl)

        y, sr = librosa.load(filename_path)
        spectral_centroid(y, sr)

columns_= ['sp_length','sp_mean','sp_median','sb_length','sb_mean','sb_median']
df2 = pd.DataFrame(data,columns=columns_)
df2['class'] = classes # Adding column with class label

 
# print(df2['class'].value_counts()) # Print the distribution of the data
# print(df2['class'].value_counts()/len(df2['class']))   
# plt.hist(df2['mean_5'], bins=20)
# plt.show()

#####################################################################################3
# Import some of the sklearn modules you are likely to use.
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold



##
feature_names = ['sp_length','sp_mean','sp_median','sb_length','sb_mean','sb_median']
class_names = ['class']


X = df2[feature_names]
y = df2[class_names]

class Debug(BaseEstimator, TransformerMixin):
    def fit(self,X,y): return self
    def transform(self,X): 
        print(X)
        return X

pipeline = Pipeline([
                #("PCA", PCA(n_components=8)),
                ("model", KNeighborsClassifier()),   
                #("debug", Debug()),     
            ])

number_of_splits = 5

metrics = [("Accuracy", accuracy_score, [])]

for train, test in StratifiedKFold(number_of_splits).split(X,y):
    pipeline.fit(X.iloc[train], y.iloc[train].values.ravel())
    predictions = pipeline.predict(X.iloc[test])
    truth = y.iloc[test]
                
    # Calculate and save the metrics for this fold
    for name, func, scores in metrics:
        score = func(truth, predictions)
        scores.append(score)

#print(pipeline['PCA'].components_)
print(scores)
