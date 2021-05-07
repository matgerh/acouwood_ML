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

def mean_features(df1, data):
    m_features = [] # Array to store features

    # Store column for easier access
    df_freq = df1['Frequency (Hz)'] 
    df_level = df1['Level (dB)']  

    RMS_all = compute_RMS(df_level) # RMS for whole spectrum
    max_all = df_level.max() # Maximum for whole spectrum
    min_all = df_level.min() # Minimum for whole spectrum
    m_features.extend([RMS_all,max_all,min_all]) # Store features in array

    # Seperate data based on 5 intervals and store in dataframes
    df_level_r1 = df_level[df_freq <= 4410] # Below or equal to 4410 Hz
    df_level_r2 = df_level[(4410 < df_freq) & (df_freq <= 8820)] # Above 4410 Hz and below or equal to 8820 Hz
    df_level_r3 = df_level[(8820 < df_freq) & (df_freq <= 13230)] # Above 8820 Hz and below or equal to 13230 Hz
    df_level_r4 = df_level[(13230 < df_freq) & (df_freq <= 17640)] # Above 13230 Hz and below or equal to 17640 Hz
    df_level_r5 = df_level[(17640 < df_freq)] # Above 17640 Hz

    
    df_list = [df_level_r1,df_level_r2,df_level_r3,df_level_r4,df_level_r5]

    for df in df_list:
        RMS = compute_RMS(df)
        MAX = df.max()
        MIN = df.min()
        m_features.extend([RMS,MAX,MIN])

    return m_features

def compute_RMS(V):
    n = len(V)
    return sqrt(sum(v*v for v in V)/n)


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
        features_list.append(mean_features(df1,data))
        #spectral(data, sr, m_features)
        classes.append(c)


columns_list = ['RMS_all', 'MAX_all','MIN_all','RMS_r1', 'MAX_r1,', 'MIN_r1',
                'RMS_r2','MAX_r2,', 'MIN_r2','RMS_r3','MAX_r3,', 'MIN_r3',
                'RMS_r4','MAX_r4,', 'MIN_r4','RMS_r5', 'MAX_r5,', 'MIN_r5'] 

df_features = pd.DataFrame(features_list, columns = columns_list) # Create dataframe and store calculated features with feature names 
df_features['class'] = classes # Add column with class label
data = df_features # Change name to indicate it is all data

# Check the proportion of "n" and "m" observations in the dependent variable target
# print(data['class'].value_counts()/len(data['class']))   
# plt.hist(data['RMS_all'], bins=20)
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
from sklearn.feature_selection import SelectKBest, f_classif


# Encode categorical target variable
le = LabelEncoder()
data['class'] = le.fit_transform(data['class'])

feature_names = columns_list # Define the name of features used
class_names = ['class'] # Define name of target variable

# Store data from df in X and y variables 
X = data[feature_names]
y = data[class_names]

# # Feature selection
# test = SelectKBest(score_func=f_classif, k='all')
# fit = test.fit(X, y)
# features = fit.transform(X)
# # Print the scores for the features
# for i in range(len(fit.scores_)):
# 	print('Feature %d: %f' % (i, fit.scores_[i]))

# # Plot the scores
# plt.bar(columns_list, fit.scores_, color="#47903A")
# ax = plt.gca()
# ax.tick_params(axis='x', labelsize=6)
# plt.title('Feature scores')
# #plt.show()

class Debug(BaseEstimator, TransformerMixin):
    def fit(self,X,y): return self
    def transform(self,X): 
        print(X)
        return X

pipeline = Pipeline([
                    ("columns", ColumnTransformer([
                        #("scaler", StandardScaler(), columns_list),
                        ], remainder="passthrough")),
                ('f_classif', SelectKBest(f_classif, k=10)),
                #("encoder", OneHotEncoder()),
                #("debug", Debug()),   
                #("PCA", PCA(n_components=8)),
                ("model", KNeighborsClassifier()),     
            ])

number_of_splits = 5

metrics = [("Accuracy", accuracy_score, []), ("Precision", precision_score, []), ("Recall", recall_score, []),]

# mlflow.log_param("number_of_splits", number_of_splits)
# mlflow.log_param("transformers", pipeline["columns"])
# mlflow.log_param("model", pipeline["model"])

# Cross validation 
for train, test in StratifiedKFold(number_of_splits).split(X,y):
    pipeline.fit(X.iloc[train], y.iloc[train].values.ravel())
    predictions = pipeline.predict(X.iloc[test])
    print(predictions)
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