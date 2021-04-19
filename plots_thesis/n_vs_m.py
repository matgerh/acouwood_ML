import os
import pandas as pd
from math import sqrt
import mlflow
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)

data = []

#Read in files
n11 = pd.read_csv('../data/1_1_n.txt', delimiter="\t")
n41 = pd.read_csv('../data/4_1_n.txt', delimiter="\t")
n12 = pd.read_csv('../data/1_2_n.txt', delimiter="\t")
n42 = pd.read_csv('../data/4_2_n.txt', delimiter="\t")
n13 = pd.read_csv('../data/1_3_n.txt', delimiter="\t")
n43 = pd.read_csv('../data/4_3_n.txt', delimiter="\t")

n14 = pd.read_csv('../data/1_4_n.txt', delimiter="\t")
n44 = pd.read_csv('../data/4_4_n.txt', delimiter="\t")
n15 = pd.read_csv('../data/1_5_n.txt', delimiter="\t")
n45 = pd.read_csv('../data/4_5_n.txt', delimiter="\t")
n16 = pd.read_csv('../data/1_6_n.txt', delimiter="\t")
n46 = pd.read_csv('../data/4_6_n.txt', delimiter="\t")

d14 = pd.read_csv('../data/1_4_d.txt', delimiter="\t")
d44 = pd.read_csv('../data/4_4_d.txt', delimiter="\t")
d15 = pd.read_csv('../data/1_5_d.txt', delimiter="\t")
d45 = pd.read_csv('../data/4_5_d.txt', delimiter="\t")
d16 = pd.read_csv('../data/1_6_d.txt', delimiter="\t")
d46 = pd.read_csv('../data/4_6_d.txt', delimiter="\t")

m17 = pd.read_csv('../data/1_7_m.txt', delimiter="\t")
m47 = pd.read_csv('../data/4_7_m.txt', delimiter="\t")
m18 = pd.read_csv('../data/1_8_m.txt', delimiter="\t")
m48 = pd.read_csv('../data/4_8_m.txt', delimiter="\t")
m19 = pd.read_csv('../data/1_9_m.txt', delimiter="\t")
m49 = pd.read_csv('../data/4_9_m.txt', delimiter="\t")

e14 = pd.read_csv('../data/1_4_e.txt', delimiter="\t")
e44 = pd.read_csv('../data/4_4_e.txt', delimiter="\t")
e15 = pd.read_csv('../data/1_5_e.txt', delimiter="\t")
e45 = pd.read_csv('../data/4_5_e.txt', delimiter="\t")
e16 = pd.read_csv('../data/1_6_e.txt', delimiter="\t")
e46 = pd.read_csv('../data/4_6_e.txt', delimiter="\t")

fig, axes = plt.subplots()
x = 'Frequency (Hz)'
y = 'Level (dB)'

a = 'n11 - m17'
b = 'n41 - m47'
c = 'n12 - m18'
d = 'n42 - m48'
e = 'n13 - m19'
f = 'n13 - m49'
g = 'n14 - m17'
h = 'n44 - m49'
i = 'n15 - m48'
j = 'n16 - m18'

rel = pd.DataFrame(data, columns=[x, a, b, c, d, e, f, g, h, i])
rel[x] = n11[x]

rel[a] = n11[y].subtract(m17[y])
rel[b] = n41[y].subtract(m47[y])
rel[c] = n12[y].subtract(m18[y])
rel[d] = n42[y].subtract(m48[y])
rel[e] = n13[y].subtract(m19[y])
rel[f] = n43[y].subtract(m49[y])
rel[g] = n14[y].subtract(m17[y])
rel[h] = n44[y].subtract(m49[y])
rel[i] = n15[y].subtract(m48[y])
rel[j] = n16[y].subtract(m18[y])

rel.plot(x, a, ax=axes, color='darkgreen')
rel.plot(x, b, ax=axes, color='yellow')
rel.plot(x, c, ax=axes, color='brown')
rel.plot(x, d, ax=axes, color='blue')
rel.plot(x, e, ax=axes, color='darkgoldenrod')
rel.plot(x, f, ax=axes, color='darkblue')
rel.plot(x, g, ax=axes, color='green')
rel.plot(x, h, ax=axes, color='yellow')
rel.plot(x, i, ax=axes, color='brown')
rel.plot(x, j, ax=axes, color='blue')


axes.set_title("Difference between samples in categories; normal and moisture")
axes.set_xlabel(x)
axes.set_ylabel(y)
#axes.set_xlim(17000,22000)
#axes.set_ylim(-15,15)
plt.legend(loc=2, prop={'size': 6})
#axes.set_xscale('log')
# axes.get_legend().remove()
#plt.savefig('diff_plots/diff_n_d_maxfreq10000.png',dpi=300)
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

