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
  "l": 0,
  "m": 1,
  "d": 2,
  "e": 2
}

data = []

#with mlflow.start_run(run_name="first_run"): 

def above18(df1):
    #should calculate features and therefore take df as parameter

    df_freq = df1['Frequency (Hz)']
    df_amp = df1['Level (dB)']
    
    mean_all = df_amp.mean()
    RMS_all = qmean(df_amp)
    mx_all = df_amp.max()
    mn_all = df_amp.min()
    return RMS_all, mean_all, mx_all, mn_all

def qmean(num):
    return sqrt(sum(n*n for n in num)/len(num))

for filename in os.listdir(dir):
    cl = classdict[filename[4:5]]
    if cl == 2:
        continue
    else:
        filename_path = os.path.join(dir, filename)
        df1 = pd.read_csv(filename_path, delimiter = "\t")
        a, b, c, d = above18(df1)
        data.append([a, b, c, d, cl]) 

df2 = pd.DataFrame(data, columns=['RMS_all', 'mean_all', 'max_all', 'min_all', 'class_all'])

print(df2)
exit()


#####################################################################################3
# Import some of the sklearn modules you are likely to use.
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV



## here we can implement different feature scenarios
feature_names = ["RMS", "mean",'max','min']
class_names = ["class"]


X = df2[feature_names]
y = df2[class_names]

class Debug(BaseEstimator, TransformerMixin):
    def fit(self,X,y): return self
    def transform(self,X): 
        print(X)
        return X

pipeline = Pipeline([
                ("scaler", KNeighborsClassifier()),   
                #("debug", Debug()),     
            ])

number_of_splits = 5


metrics = [("Accuracy", accuracy_score, [])]

for train, test in TimeSeriesSplit(number_of_splits).split(X,y):
    pipeline.fit(X.iloc[train], y.iloc[train].values.ravel())
    predictions = pipeline.predict(X.iloc[test])
    truth = y.iloc[test]
                
    # Calculate and save the metrics for this fold
    for name, func, scores in metrics:
        score = func(truth, predictions)
        scores.append(score)

print(scores)
