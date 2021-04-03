import os
import pandas as pd
from math import sqrt

dir = 'data/'

classdict = {
  "n": 0,
  "l": 1,
  "m": 2,
  "d": 3,
  "e": 4
}

data = []

def myfunc(df1):
    #should calculate features and therefore take df as parameter
    outputdB = df1['Level (dB)']
    RMS = qmean(outputdB)
    
    df2 = df1['Level (dB)'][df1['Frequency (Hz)'] > 22000]
    avg = df2.mean()



    f = 4
    return RMS, avg, f

def qmean(num):
	return sqrt(sum(n*n for n in num)/len(num))



#def classfunc():

    #this function should read label from filename and return label
    # what classes 0,1,2? should be readable to tensorflow 
    # make a lookup tabel with filename and class number 

for filename in os.listdir(dir):
    if filename.endswith(".txt"):
        c = classdict[filename[4:5]]

        filename_path = os.path.join(dir, filename)
        df1 = pd.read_csv(filename_path, delimiter = "\t")
        #below function should take df1 as parameter 
        a, b, d = myfunc(df1)
        data.append([a, b, d])  

        print(filename)

df2 = pd.DataFrame(data, columns=['RMS', 'avg', 'class'])

print(df2)
