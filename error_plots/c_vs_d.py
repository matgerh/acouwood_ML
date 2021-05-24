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

d14 = pd.read_csv('../fft_data/1_4_d.txt', delimiter="\t")
d44 = pd.read_csv('../fft_data/4_4_d.txt', delimiter="\t")
d15 = pd.read_csv('../fft_data/1_5_d.txt', delimiter="\t")
d45 = pd.read_csv('../fft_data/4_5_d.txt', delimiter="\t")
d16 = pd.read_csv('../fft_data/1_6_d.txt', delimiter="\t")
d46 = pd.read_csv('../fft_data/4_6_d.txt', delimiter="\t")

m17 = pd.read_csv('../fft_data/1_7_m.txt', delimiter="\t")
m47 = pd.read_csv('../fft_data/4_7_m.txt', delimiter="\t")
m18 = pd.read_csv('../fft_data/1_8_m.txt', delimiter="\t")
m48 = pd.read_csv('../fft_data/4_8_m.txt', delimiter="\t")
m19 = pd.read_csv('../fft_data/1_9_m.txt', delimiter="\t")
m49 = pd.read_csv('../fft_data/4_9_m.txt', delimiter="\t")

e14 = pd.read_csv('../fft_data/1_4_e.txt', delimiter="\t")
e44 = pd.read_csv('../fft_data/4_4_e.txt', delimiter="\t")
e15 = pd.read_csv('../fft_data/1_5_e.txt', delimiter="\t")
e45 = pd.read_csv('../fft_data/4_5_e.txt', delimiter="\t")
e16 = pd.read_csv('../fft_data/1_6_e.txt', delimiter="\t")
e46 = pd.read_csv('../fft_data/4_6_e.txt', delimiter="\t")

fig, axes = plt.subplots()
x = 'Frequency (Hz)'
y = 'Level (dB)'

a = 'c11 - d14'
b = 'c41 - d44'
c = 'c12 - d15'
d = 'c42 - d45'
e = 'c13 - d16'
f = 'c13 - d46'
g = 'c14 - d14'
h = 'c44 - d44'
i = 'c15 - d15'
j = 'c16 - d45'

rel = pd.DataFrame(data, columns=[x, a, b, c, d, e, f, g, h, i])
rel[x] = c11[x]

rel[a] = c11[y].subtract(d14[y])**2
rel[b] = c41[y].subtract(d44[y])**2
rel[c] = c12[y].subtract(d15[y])**2
rel[d] = c42[y].subtract(d45[y])**2
rel[e] = c13[y].subtract(d16[y])**2
rel[f] = c43[y].subtract(d46[y])**2

rel[g] = c14[y].subtract(d14[y])**2
rel[h] = c44[y].subtract(d44[y])**2
rel[i] = c15[y].subtract(d15[y])**2
rel[j] = c16[y].subtract(d45[y])**2

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

#axes.set_title("Squared error between control and decay samples")
axes.set_xlabel(x)
axes.set_ylabel(y)
#axes.set_xlim(20,22000)
#axes.set_ylim(-15,15)
plt.legend(loc=2, prop={'size': 6})
#axes.set_xscale('log')
# axes.get_legend().remove()
plt.savefig('c_vs_d.png',dpi=300)
plt.show()
exit()


