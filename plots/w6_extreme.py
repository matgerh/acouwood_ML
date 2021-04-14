import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

#Read in files
n11 = pd.read_csv('../data/1_6_e.txt', delimiter="\t")
n21 = pd.read_csv('../data/2_6_e.txt', delimiter="\t")
n31 = pd.read_csv('../data/3_6_e.txt', delimiter="\t")
n41 = pd.read_csv('../data/4_6_e.txt', delimiter="\t")
n51 = pd.read_csv('../data/5_6_e.txt', delimiter="\t")
n61 = pd.read_csv('../data/6_6_e.txt', delimiter="\t")

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

axes.set_title("sample 6 - extreme moisture")
axes.set_xlabel(x)
axes.set_ylabel(y)
#axes.set_xscale('log')
axes.get_legend().remove()

plt.show()