import os
import pandas as pd
from math import sqrt
import mlflow
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)

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

a = 'n11 - d14'
b = 'n41 - d44'
c = 'n12 - d15'
d = 'n42 - d45'
e = 'n13 - d16'
f = 'n13 - d46'
g = 'n14 - d14'
h = 'n44 - d44'
i = 'n15 - d15'
j = 'n16 - d45'

rel = pd.DataFrame(data, columns=[x, a, b, c, d, e, f, g, h, i])
rel[x] = n11[x]

rel[a] = n11[y].subtract(d14[y])**2
rel[b] = n41[y].subtract(d44[y])**2
rel[c] = n12[y].subtract(d15[y])**2
rel[d] = n42[y].subtract(d45[y])**2
rel[e] = n13[y].subtract(d16[y])**2
rel[f] = n43[y].subtract(d46[y])**2

rel[g] = n14[y].subtract(d14[y])**2
rel[h] = n44[y].subtract(d44[y])**2
rel[i] = n15[y].subtract(d15[y])**2
rel[j] = n16[y].subtract(d45[y])**2

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

axes.set_title("Squared error between neutral and decay samples")
axes.set_xlabel(x)
axes.set_ylabel(y)
axes.set_xlim(20,22000)
#axes.set_ylim(-15,15)
plt.legend(loc=2, prop={'size': 6})
#axes.set_xscale('log')
# axes.get_legend().remove()
#plt.savefig('diff_plots/diff_n_d_maxfreq10000.png',dpi=300)
plt.show()
exit()


