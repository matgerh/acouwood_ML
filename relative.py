import os
import pandas as pd
from math import sqrt
import mlflow
import matplotlib.pyplot as plt

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




#Read in files
n11 = pd.read_csv('data/1_1_n.txt', delimiter="\t")
n41 = pd.read_csv('data/4_1_n.txt', delimiter="\t")
n12 = pd.read_csv('data/1_2_n.txt', delimiter="\t")
n42 = pd.read_csv('data/4_2_n.txt', delimiter="\t")
n13 = pd.read_csv('data/1_3_n.txt', delimiter="\t")
n43 = pd.read_csv('data/4_3_n.txt', delimiter="\t")

d14 = pd.read_csv('data/1_4_d.txt', delimiter="\t")
d44 = pd.read_csv('data/4_4_d.txt', delimiter="\t")
d15 = pd.read_csv('data/1_5_d.txt', delimiter="\t")
d45 = pd.read_csv('data/4_5_d.txt', delimiter="\t")
d16 = pd.read_csv('data/1_6_d.txt', delimiter="\t")
d46 = pd.read_csv('data/4_6_d.txt', delimiter="\t")

l17 = pd.read_csv('data/1_7_l.txt', delimiter="\t")
l47 = pd.read_csv('data/4_7_l.txt', delimiter="\t")
l18 = pd.read_csv('data/1_8_l.txt', delimiter="\t")
l48 = pd.read_csv('data/4_8_l.txt', delimiter="\t")
l19 = pd.read_csv('data/1_9_l.txt', delimiter="\t")
l49 = pd.read_csv('data/4_9_l.txt', delimiter="\t")

m17 = pd.read_csv('data/1_7_m.txt', delimiter="\t")
m47 = pd.read_csv('data/4_7_m.txt', delimiter="\t")
m18 = pd.read_csv('data/1_8_m.txt', delimiter="\t")
m48 = pd.read_csv('data/4_8_m.txt', delimiter="\t")
m19 = pd.read_csv('data/1_9_m.txt', delimiter="\t")
m49 = pd.read_csv('data/4_9_m.txt', delimiter="\t")

e14 = pd.read_csv('data/1_4_e.txt', delimiter="\t")
e44 = pd.read_csv('data/4_4_e.txt', delimiter="\t")
e15 = pd.read_csv('data/1_5_e.txt', delimiter="\t")
e45 = pd.read_csv('data/4_5_e.txt', delimiter="\t")
e16 = pd.read_csv('data/1_6_e.txt', delimiter="\t")
e46 = pd.read_csv('data/4_6_e.txt', delimiter="\t")

fig, axes = plt.subplots()
x = 'Frequency (Hz)'
y = 'Level (dB)'


a = 'n21 - m1'
b = 'n41 - m4'
c = 'n61 - m3'
d = 'n31 - e5'
e = 'n51 - e2'
f = 'n11 - e6'

rel = pd.DataFrame(data, columns=[x, a, b, c, d, e])
rel[x] = n11[x]

rel[a] = n21[y].subtract(m1[y])
rel[b] = n41[y].subtract(m4[y])
rel[c] = n61[y].subtract(m3[y])
rel[d] = n31[y].subtract(e5[y])
rel[e] = n51[y].subtract(e2[y])
rel[f] = n11[y].subtract(e6[y])


rel.plot(x, a, ax=axes, color='darkgreen')
rel.plot(x, b, ax=axes, color='darkgreen')
rel.plot(x, c, ax=axes, color='darkgreen')
rel.plot(x, d, ax=axes, color='blue')
rel.plot(x, e, ax=axes, color='blue')
rel.plot(x, f, ax=axes, color='blue')

axes.set_title("relative")
axes.set_xlabel(x)
axes.set_ylabel(y)
axes.set_xscale('log')
# axes.get_legend().remove()
plt.show()
exit()









# dirn = 'data/n'
# dird = 'data/d'
# dirm = 'data/m'


# i = 0
# for filename in os.listdir(dirn)[:6]:
#     i = i+1
#     if filename.endswith(".txt"):
#         c = classdict[filename[4:5]]
#         filename_path = os.path.join(dirn, filename)
#         dfn[f'{c}{i}'] = pd.read_csv(filename_path, delimiter = "\t")
# print(dfn)

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

