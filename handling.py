import os
import pandas as pd

dir = 'data/'

data = []

def myfunc():
    #should calculate features and therefore take df as parameter
    d = 2
    e = 3
    f = 4
    return d, e, f

for filename in os.listdir(dir):
    if filename.endswith(".txt"):

        filename_path = os.path.join(dir, filename)
        df1 = pd.read_csv(filename_path, delimiter = "\t")
        #below function should take df1 as parameter 
        a, b, c = myfunc()
        data.append([a, b, c])  

        print(filename)

df2 = pd.DataFrame(data, columns=['A', 'B', 'C'])

print(df2)
