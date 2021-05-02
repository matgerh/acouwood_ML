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

#mlflow.set_tracking_uri("http://localhost:5000/")


# ML experiment name
#mlflow.set_experiment("AcouWood")

pd.set_option('display.max_rows', None)

features_list = []
classes = []

#with mlflow.start_run(run_name="KNN w. Standardscaler"): 

def mean_features(df1):
    m_features = [] # Array to store features
    df_freq = df1['Frequency (Hz)']
    df_amp = df1['Level (dB)']
    RMS_overall = compute_RMS(df_amp) # Compute RMS for whole spectrum
    m_features.append(RMS_overall) # Store mean in features array

    df_amp_r1 = df_amp[df_freq <= 4410] # Ampltitude data only below 4410 Hz
    df_amp_r2 = df_amp[(4410 < df_freq) & (df_freq <= 8820)] 
    df_amp_r3 = df_amp[(8820 < df_freq) & (df_freq <= 13230)]
    df_amp_r4 = df_amp[(13230 < df_freq) & (df_freq <= 17640)]
    df_amp_r5 = df_amp[(17640 < df_freq)]

    df_list = [df_amp_r1,df_amp_r2,df_amp_r3,df_amp_r4,df_amp_r5]

    for df in df_list:
        RMS = compute_RMS(df)
        m_features.append((RMS))

    return m_features

def compute_RMS(V):
    n = len(V)
    return sqrt(sum(v*v for v in V)/n)

# def spectral(y, sr, m_features):
#     sp = librosa.feature.spectral_centroid(y=y, sr=sr)[0] # Compute all centroid frequencies and store in array
#     sp_length = len(sp) 
#     sp_mean = np.mean(sp)
#     sp_median = np.median(sp)

#     sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0] # Compute all centroid frequencies and store in array
#     sb_length = len(sb) 
#     sb_mean = np.mean(sb)
#     sb_median = np.median(sb)

#     s_features = sp_length,sp_mean,sp_median,sb_length,sb_mean,sb_median
#     m_features.extend(s_features)
#     features_list.append(m_features)

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
    # print(df1)
    # plt.plot(freq, MAG)
    # plt.show()
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
        features_list.append(mean_features(df1))
        #spectral(data, sr, m_features)
        classes.append(c)

# 'sp_length','sp_mean','sp_median','sb_length','sb_mean','sb_median',
columns_list = ['mean_all', 'RMS_mean','mean_r1','mean_r2','mean_r3','mean_r4','mean_r5']

df_features = pd.DataFrame(features_list, columns = columns_list) # Create dataframe and store calculated features with feature names 
df_features['class'] = classes # Add column with class label
data = df_features # Change name to indicate it is all data

print(data)

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
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


# Encode categorical target variable
le = LabelEncoder()
data['class'] = le.fit_transform(data['class'])

feature_names = columns_list # Define the name of features used
class_names = ['class'] # Define name of target variable

# Store data from df in X and y variables 
X = data[feature_names]
y = data[class_names]

# Feature selection
test = SelectKBest(score_func=f_classif, k='all')
fit = test.fit(X, y)
features = fit.transform(X)
# Print the scores for the features
for i in range(len(fit.scores_)):
	print('Feature %d: %f' % (i, fit.scores_[i]))
# Plot the scores
# plt.bar([i for i in range(len(fit.scores_))], fit.scores_)
# plt.show()

class Debug(BaseEstimator, TransformerMixin):
    def fit(self,X,y): return self
    def transform(self,X): 
        print(X)
        return X

pipeline = Pipeline([
                    ("columns", ColumnTransformer([
                        ("scaler", StandardScaler(), columns_list),
                        ], remainder="passthrough")),
                #("encoder", OneHotEncoder()),
                #("debug", Debug()),   
                #("PCA", PCA(n_components=8)),
                ("model", KNeighborsClassifier()),     
            ])

number_of_splits = 5

metrics = [("Accuracy", accuracy_score, []), ("Precision", precision_score, []), ("Recall", recall_score, [])]

# mlflow.log_param("number_of_splits", number_of_splits)
# mlflow.log_param("transformers", pipeline["columns"])
# mlflow.log_param("model", pipeline["model"])

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
    # mlflow.log_metric(f"mean_{name}", mean_score)
    # mlflow.log_metric(f"std_{name}", std_score)  
    print(f"mean_{name}", mean_score)  

#print(pipeline['PCA'].components_)