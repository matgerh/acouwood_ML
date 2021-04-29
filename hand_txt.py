import os
import pandas as pd
from math import sqrt
import mlflow
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import r2_score


# ML experiment name
#mlflow.set_experiment("AcouWood")

pd.set_option('display.max_rows', None)

dir = 'fft_data/'

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

for filename in os.listdir(dir):
    cl = classdict[filename[4:5]]
    if cl == 2:
        continue
    else:
        filename_path = os.path.join(dir, filename)
        df1 = pd.read_csv(filename_path, delimiter = "\t")
        print(df1)
        exit()
        data = above18(df1)
        classes.append(cl)
        
columns_= ['mean_5', 'max_5', 'min_5', 'mean_5_10', 'max_5_10', 
          'min_5_10', 'mean_10_15', 'max_10_15', 'min_10_15', 
          'mean_15_20', 'max_15_20','min_15_20','mean_20','max_20','min_20']

df2 = pd.DataFrame(data,columns=columns_)
df2['class'] = classes # Adding column with class label
print(df2['class'].value_counts()) # Print the distribution of the data
print(df2['class'].value_counts()/len(df2['class']))   
plt.hist(df2['mean_5'], bins=20)
plt.show()

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
feature_names = ['mean_5', 'max_5', 'min_5', 'mean_5_10', 'max_5_10', 
          'min_5_10', 'mean_10_15', 'max_10_15', 'min_10_15', 
          'mean_15_20', 'max_15_20','min_15_20','mean_20','max_20','min_20']
class_names = ['class']


X = df2[feature_names]
y = df2[class_names]

class Debug(BaseEstimator, TransformerMixin):
    def fit(self,X,y): return self
    def transform(self,X): 
        print(X)
        return X

pipeline = Pipeline([
                ("PCA", PCA(n_components=8)),
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
