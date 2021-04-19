import pandas as pd
from numpy import mean
from numpy import std
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import numpy as np

#Read in files
n11 = pd.read_csv('../data/1_1_n.txt', delimiter="\t")
n21 = pd.read_csv('../data/2_1_n.txt', delimiter="\t")
n31 = pd.read_csv('../data/3_1_n.txt', delimiter="\t")
n41 = pd.read_csv('../data/4_1_n.txt', delimiter="\t")
n51 = pd.read_csv('../data/5_1_n.txt', delimiter="\t")
n61 = pd.read_csv('../data/6_1_n.txt', delimiter="\t")

fig, axes = plt.subplots()
x = 'Frequency (Hz)'
y = 'Level (dB)'

# Normal sample 1
n11.plot(x, y, ax=axes, color='darkgreen')
n21.plot(x, y, ax=axes, color='darkgreen')
n31.plot(x, y, ax=axes, color='darkgreen')
n41.plot(x, y, ax=axes, color='blue')
n51.plot(x, y, ax=axes, color='blue')
n61.plot(x, y, ax=axes, color='blue')

axes.set_title("sample 1 - normal")
axes.set_xlabel(x)
axes.set_ylabel(y)
#axes.set_xscale('log')
axes.get_legend().remove()

print('data1: mean=%.3f stdv=%.3f' % (n11[y].mean(), n11[y].std()))
print('data2: mean=%.3f stdv=%.3f' % (n21[y].mean(), n21[y].std()))
print('data3: mean=%.3f stdv=%.3f' % (n31[y].mean(), n31[y].std()))
print('data4: mean=%.3f stdv=%.3f' % (n41[y].mean(), n41[y].std()))
print('data5: mean=%.3f stdv=%.3f' % (n51[y].mean(), n51[y].std()))
print('data6: mean=%.3f stdv=%.3f' % (n61[y].mean(), n61[y].std()))

corr1, _ = pearsonr(n11[y], n21[y])
corr2, _ = pearsonr(n11[y], n31[y])
corr3, _ = pearsonr(n41[y], n51[y])
corr4, _ = pearsonr(n61[y], n51[y])
corr5, _ = pearsonr(n41[y], n61[y])
print('Pearsons correlation: %.3f' % corr1)
print('Pearsons correlation: %.3f' % corr2)
print('Pearsons correlation: %.3f' % corr3)
print('Pearsons correlation: %.3f' % corr4)
print('Pearsons correlation: %.3f' % corr5)

plt.show()
