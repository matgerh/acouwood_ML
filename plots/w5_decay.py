import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

#Read in files
n11 = pd.read_csv('../data/1_5_d.txt', delimiter="\t")
n21 = pd.read_csv('../data/2_5_d.txt', delimiter="\t")
n31 = pd.read_csv('../data/3_5_d.txt', delimiter="\t")
n41 = pd.read_csv('../data/4_5_d.txt', delimiter="\t")
n51 = pd.read_csv('../data/5_5_d.txt', delimiter="\t")
n61 = pd.read_csv('../data/6_5_d.txt', delimiter="\t")

fig, axes = plt.subplots()
x = 'Frequency (Hz)'
y = 'Level (dB)'

# Normal sample 1
n11.plot(x, y, ax=axes, color='orange')
n21.plot(x, y, ax=axes, color='orange')
n31.plot(x, y, ax=axes, color='orange')
n41.plot(x, y, ax=axes, color='#FBD255')
n51.plot(x, y, ax=axes, color='#FBD255')
n61.plot(x, y, ax=axes, color='#FBD255')

axes.set_title("sample 5 - decay")
axes.set_xlabel(x)
axes.set_ylabel(y)
#axes.set_xscale('log')
axes.get_legend().remove()

plt.show()