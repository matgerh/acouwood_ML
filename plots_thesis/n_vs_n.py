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


fig, axes = plt.subplots()
x = 'Frequency (Hz)'
y = 'Level (dB)'

a = 'n11 - n14'
b = 'n41 - n44'
c = 'n12 - n15'
d = 'n42 - n45'
e = 'n13 - n16'
f = 'n13 - n46'

rel = pd.DataFrame(data, columns=[x, a, b, c, d, e])
rel[x] = n11[x]

rel[a] = n11[y].subtract(n14[y])
rel[b] = n41[y].subtract(n44[y])
rel[c] = n12[y].subtract(n15[y])
rel[d] = n42[y].subtract(n45[y])
rel[e] = n13[y].subtract(n16[y])
rel[f] = n43[y].subtract(n46[y])

rel.plot(x, a, ax=axes, color='darkgreen')
rel.plot(x, b, ax=axes, color='yellow')
rel.plot(x, c, ax=axes, color='brown')
rel.plot(x, d, ax=axes, color='blue')
rel.plot(x, e, ax=axes, color='darkgoldenrod')
rel.plot(x, f, ax=axes, color='darkblue')

axes.set_title("Difference between samples in categories; normal and normal")
axes.set_xlabel(x)
axes.set_ylabel(y)
axes.set_xlim(0,22000)
#axes.set_ylim(-20,20)
#axes.set_xscale('log')
# axes.get_legend().remove()
#plt.savefig('diff_plots/diff_n_d_maxfreq10000.png',dpi=300)
plt.show()
exit()

