import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

#Read in files
n1 = pd.read_csv('../data/1_1_n.txt', delimiter="\t")
n2 = pd.read_csv('../data/6_1_n.txt', delimiter="\t")

fig, axes = plt.subplots(1, sharex=True)
x = 'Frequency (Hz)'
y = 'Level (dB)'

y1 = n1['Level (dB)']
y2 = n2['Level (dB)']

# Normal sample 1
n1.plot(x, y, ax=axes, color='darkgreen')
n2.plot(x, y, ax=axes, color='darkgreen')

axes.set_title("sample 1 - normal")
axes.set_xlabel(x)
axes.set_ylabel(y)
#axes.set_xscale('log')
axes.get_legend().remove()
axes.fill_between(x, y1, y2, where=y2 >= y1, color="red", interpolate=True)

plt.show()
