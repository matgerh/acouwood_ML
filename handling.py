import os
import pandas as pd
from math import sqrt

dir = 'data/'

data = []

def myfunc(df1):
    #should calculate features and therefore take df as parameter
    outputdB = df1['Level (dB)']
    RMS = qmean(outputdB)

    e = 3
    f = 4
    return RMS, e, f

def qmean(num):
	return sqrt(sum(n*n for n in num)/len(num))



def classfunc():
    #this function should read label from filename and return label
    # what classes 0,1,2? should be readable to tensorflow 
    # make a lookup tabel with filename and class number 

for filename in os.listdir(dir):
    if filename.endswith(".txt"):

        filename_path = os.path.join(dir, filename)
        df1 = pd.read_csv(filename_path, delimiter = "\t")
        #below function should take df1 as parameter 
        a, b, c = myfunc(df1)
        data.append([a, b, c])  

        print(filename)

df2 = pd.DataFrame(data, columns=['A', 'B', 'C'])

print(df2)
