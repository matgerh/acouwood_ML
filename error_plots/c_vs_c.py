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
c11 = pd.read_csv('../fft_data/1_1_c.txt', delimiter="\t")
c41 = pd.read_csv('../fft_data/4_1_c.txt', delimiter="\t")
c12 = pd.read_csv('../fft_data/1_2_c.txt', delimiter="\t")
c42 = pd.read_csv('../fft_data/4_2_c.txt', delimiter="\t")
c13 = pd.read_csv('../fft_data/1_3_c.txt', delimiter="\t")
c43 = pd.read_csv('../fft_data/4_3_c.txt', delimiter="\t")

c14 = pd.read_csv('../fft_data/1_4_c.txt', delimiter="\t")
c44 = pd.read_csv('../fft_data/4_4_c.txt', delimiter="\t")
c15 = pd.read_csv('../fft_data/1_5_c.txt', delimiter="\t")
c45 = pd.read_csv('../fft_data/4_5_c.txt', delimiter="\t")
c16 = pd.read_csv('../fft_data/1_6_c.txt', delimiter="\t")
c46 = pd.read_csv('../fft_data/4_6_c.txt', delimiter="\t")

c17 = pd.read_csv('../fft_data/1_7_c.txt', delimiter="\t")
c47 = pd.read_csv('../fft_data/4_7_c.txt', delimiter="\t")
c18 = pd.read_csv('../fft_data/1_8_c.txt', delimiter="\t")
c48 = pd.read_csv('../fft_data/4_8_c.txt', delimiter="\t")
c19 = pd.read_csv('../fft_data/1_9_c.txt', delimiter="\t")
c49 = pd.read_csv('../fft_data/4_9_c.txt', delimiter="\t")


fig, axes = plt.subplots()
x = 'Frequency (Hz)'
y = 'Level (dB)'

a = 'c11 - c14'
b = 'c41 - c44'
c = 'c12 - c15'
d = 'c42 - c45'
e = 'c13 - c16'
f = 'c13 - c46'

g = 'c47 - c41'
h = 'c18 - c15'
i = 'c49 - c46'
j = 'c48 - c47'

rel = pd.DataFrame(data, columns=[x, a, b, c, d, e])
rel[x] = c11[x]

rel[a] = c11[y].subtract(c14[y])**2
rel[b] = c41[y].subtract(c44[y])**2
rel[c] = c12[y].subtract(c15[y])**2
rel[d] = c42[y].subtract(c45[y])**2
rel[e] = c13[y].subtract(c16[y])**2
rel[f] = c43[y].subtract(c46[y])**2

rel[g] = c47[y].subtract(c41[y])**2
rel[h] = c18[y].subtract(c15[y])**2
rel[i] = c49[y].subtract(c46[y])**2
rel[j] = c48[y].subtract(c47[y])**2


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

axes.set_title("Squared error between control and control samples")
axes.set_xlabel(x)
axes.set_ylabel(y)
axes.set_xlim(20,22000)
#axes.set_ylim(-20,20)
#axes.set_xscale('log')
# axes.get_legend().remove()
plt.savefig('c_vs_c.png',dpi=300)
plt.show()
exit()

