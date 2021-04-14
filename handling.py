import os
import pandas as pd
from math import sqrt
import mlflow


# Set the experiment name
mlflow.set_experiment("AcouWood")

pd.set_option('display.max_rows', None)

dir = 'data/'

classdict = {
  "n": 0,
  "l": 0,
  "m": 1,
  "d": 1,
  "e": 1
}

data = []

def myfunc(df1):
    #should calculate features and therefore take df as parameter
    outputdB = df1['Level (dB)']
    RMS = qmean(outputdB)
    return RMS

def qmean(num):
	return sqrt(sum(n*n for n in num)/len(num))

for filename in os.listdir(dir):
    if filename.endswith(".txt"):
        c = classdict[filename[4:5]]
        filename_path = os.path.join(dir, filename)
        df1 = pd.read_csv(filename_path, delimiter = "\t")
        df2 = [df1['Frequency (Hz)'] > 18000] # Data only above 18 khz

        a = myfunc(df1)
        data.append([a, c]) 

df2 = pd.DataFrame(data, columns=['RMS', 'class'])


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
feature_names = ["RMS"]
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
