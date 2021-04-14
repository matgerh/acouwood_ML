import os
import pandas as pd
from math import sqrt
import mlflow


pd.set_option('display.max_rows', None)

dir = 'data/'

classdict = {
  "n": "n",
  "l": "n",
  "m": "m",
  "d": "d",
  "e": "d"
}

data = []



dirn = 'data/n'
dird = 'data/d'
dirm = 'data/m'

dfn = pd.Dataframe
dfd = None
dfm = None

i = 0
for filename in os.listdir(dirn)[:6]:
    i = i+1
    if filename.endswith(".txt"):
        c = classdict[filename[4:5]]
        filename_path = os.path.join(dirn, filename)
        dfn[f'{c}{i}'] = pd.read_csv(filename_path, delimiter = "\t")
print(dfn)

#         for filename in os.listdir(dird)[:6]:
#             if filename.endswith(".txt"):
#                 c = classdict[filename[4:5]]
#                 filename_path = os.path.join(dir, filename)
#                 dfd = pd.read_csv(filename_path, delimiter = "\t")

#         for filename in os.listdir(dir):
#             if filename.endswith(".txt"):
#                 c = classdict[filename[4:5]]
#                 filename_path = os.path.join(dir, filename)
#                 dfm = pd.read_csv(filename_path, delimiter = "\t")

#         n_minus_d = dfn.subtract(dfd)
#         n_minus_



        




#         a = myfunc(df1)
#         data.append([a, c]) 

# df2 = pd.DataFrame(data, columns=['RMS', 'class'])

