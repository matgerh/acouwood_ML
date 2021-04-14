import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

#Read in files
n11 = pd.read_csv('../data/1_9_m.txt', delimiter="\t")
n21 = pd.read_csv('../data/2_9_m.txt', delimiter="\t")
n31 = pd.read_csv('../data/3_9_m.txt', delimiter="\t")
n41 = pd.read_csv('../data/4_9_m.txt', delimiter="\t")
n51 = pd.read_csv('../data/5_9_m.txt', delimiter="\t")
n61 = pd.read_csv('../data/6_9_m.txt', delimiter="\t")

fig, axes = plt.subplots()
x = 'Frequency (Hz)'
y = 'Level (dB)'

# Normal sample 1
n11.plot(x, y, ax=axes, color='#1878D5')
n21.plot(x, y, ax=axes, color='#1878D5')
n31.plot(x, y, ax=axes, color='#1878D5')
n41.plot(x, y, ax=axes, color='#6593D8')
n51.plot(x, y, ax=axes, color='#6593D8')
n61.plot(x, y, ax=axes, color='#6593D8')

axes.set_title("sample 9 - moisture")
axes.set_xlabel(x)
axes.set_ylabel(y)
#axes.set_xscale('log')
axes.get_legend().remove()

plt.show()