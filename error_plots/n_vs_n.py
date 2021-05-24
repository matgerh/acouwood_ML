import os
import pandas as pd
from math import sqrt
import mlflow
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)

c1 = 'limegreen'
c2 = 'steelblue'
c3 = 'deepskyblue'
c4 = 'sienna'
c5 = 'darkgoldenrod'
c6 = 'darkgreen'
c7 = 'peru'
c8 = 'darkkhaki'
c9 = 'royalblue'
c10 = 'yellowgreen'

data = []

#Read in files
n11 = pd.read_csv('../fft_data/1_1_n.txt', delimiter="\t")
n41 = pd.read_csv('../fft_data/4_1_n.txt', delimiter="\t")
n12 = pd.read_csv('../fft_data/1_2_n.txt', delimiter="\t")
n42 = pd.read_csv('../fft_data/4_2_n.txt', delimiter="\t")
n13 = pd.read_csv('../fft_data/1_3_n.txt', delimiter="\t")
n43 = pd.read_csv('../fft_data/4_3_n.txt', delimiter="\t")

n14 = pd.read_csv('../fft_data/1_4_n.txt', delimiter="\t")
n44 = pd.read_csv('../fft_data/4_4_n.txt', delimiter="\t")
n15 = pd.read_csv('../fft_data/1_5_n.txt', delimiter="\t")
n45 = pd.read_csv('../fft_data/4_5_n.txt', delimiter="\t")
n16 = pd.read_csv('../fft_data/1_6_n.txt', delimiter="\t")
n46 = pd.read_csv('../fft_data/4_6_n.txt', delimiter="\t")

n17 = pd.read_csv('../fft_data/1_7_n.txt', delimiter="\t")
n47 = pd.read_csv('../fft_data/4_7_n.txt', delimiter="\t")
n18 = pd.read_csv('../fft_data/1_8_n.txt', delimiter="\t")
n48 = pd.read_csv('../fft_data/4_8_n.txt', delimiter="\t")
n19 = pd.read_csv('../fft_data/1_9_n.txt', delimiter="\t")
n49 = pd.read_csv('../fft_data/4_9_n.txt', delimiter="\t")


fig, axes = plt.subplots()
x = 'Frequency (Hz)'
y = 'Level (dB)'

a = 'n11 - n14'
b = 'n41 - n44'
c = 'n12 - n15'
d = 'n42 - n45'
e = 'n13 - n16'
f = 'n13 - n46'

g = 'n47 - n41'
h = 'n18 - n15'
i = 'n49 - n46'
j = 'n48 - n47'

rel = pd.DataFrame(data, columns=[x, a, b, c, d, e])
rel[x] = n11[x]

rel[a] = n11[y].subtract(n14[y])**2
rel[b] = n41[y].subtract(n44[y])**2
rel[c] = n12[y].subtract(n15[y])**2
rel[d] = n42[y].subtract(n45[y])**2
rel[e] = n13[y].subtract(n16[y])**2
rel[f] = n43[y].subtract(n46[y])**2

rel[g] = n47[y].subtract(n41[y])**2
rel[h] = n18[y].subtract(n15[y])**2
rel[i] = n49[y].subtract(n46[y])**2
rel[j] = n48[y].subtract(n47[y])**2


rel[a] = rel[a].rolling(100, min_periods=1).sum()
rel[b] = rel[b].rolling(100, min_periods=1).sum()
rel[c] = rel[c].rolling(100, min_periods=1).sum()
rel[d] = rel[d].rolling(100, min_periods=1).sum()
rel[e] = rel[e].rolling(100, min_periods=1).sum()
rel[f] = rel[f].rolling(100, min_periods=1).sum()
rel[g] = rel[g].rolling(100, min_periods=1).sum()
rel[h] = rel[h].rolling(100, min_periods=1).sum()
rel[i] = rel[i].rolling(100, min_periods=1).sum()
rel[j] = rel[j].rolling(100, min_periods=1).sum()

rel.plot(x, a, ax=axes, color=c1)
rel.plot(x, b, ax=axes, color=c2)
rel.plot(x, c, ax=axes, color=c3)
rel.plot(x, d, ax=axes, color=c4)
rel.plot(x, e, ax=axes, color=c5)
rel.plot(x, f, ax=axes, color=c6)
rel.plot(x, g, ax=axes, color=c7)
rel.plot(x, h, ax=axes, color=c8)
rel.plot(x, i, ax=axes, color=c9)
rel.plot(x, j, ax=axes, color=c10)

axes.fill_between(rel[x], rel[a], color=c1, alpha=0.4)
axes.fill_between(rel[x], rel[b], color=c2, alpha=0.4)
axes.fill_between(rel[x], rel[c],  color=c3, alpha=0.4)
axes.fill_between(rel[x], rel[d], color=c4, alpha=0.4)
axes.fill_between(rel[x], rel[e], color=c5, alpha=0.4)
axes.fill_between(rel[x], rel[f], color=c6, alpha=0.4)
axes.fill_between(rel[x], rel[g],  color=c7, alpha=0.4)
axes.fill_between(rel[x], rel[h], color=c8, alpha=0.4)
axes.fill_between(rel[x], rel[i],  color=c9, alpha=0.4)
axes.fill_between(rel[x], rel[j], color=c10, alpha=0.4)

axes.set_title("Squared error between neutral and neutral samples")
axes.set_xlabel(x)
axes.set_ylabel(y)
axes.set_xlim(20,22000)
#axes.set_ylim(-20,20)
#axes.set_xscale('log')
# axes.get_legend().remove()
plt.savefig('error_plots/img_data/n_vs_n.png',dpi=300)
plt.show()
exit()

