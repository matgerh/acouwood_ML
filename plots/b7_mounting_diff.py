import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

#Read in files
n11 = pd.read_csv('../fft_data/1_7_m.txt', delimiter="\t")
n21 = pd.read_csv('../fft_data/2_7_m.txt', delimiter="\t")
n31 = pd.read_csv('../fft_data/3_7_m.txt', delimiter="\t")
n41 = pd.read_csv('../fft_data/4_7_m.txt', delimiter="\t")
n51 = pd.read_csv('../fft_data/5_7_m.txt', delimiter="\t")
n61 = pd.read_csv('../fft_data/6_7_m.txt', delimiter="\t")

fig, axes = plt.subplots()
x = 'Frequency (Hz)'
y = 'Level (dB)'

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

normal = "#47903A"
decay = "#E3A541"
extremeDecay = "#A2512D"
moisture = "#1878D5"
lightMoisture = "#30A1FB"

# Normal sample 1
n11.plot(x, y, ax=axes, color=c4, label='1_7_m')
n21.plot(x, y, ax=axes, color=c4, label='2_7_m')
n31.plot(x, y, ax=axes, color=c4, label='3_7_m')
n41.plot(x, y, ax=axes, color=c8, label='4_7_m')
n51.plot(x, y, ax=axes, color=c8, label='5_7_m')
n61.plot(x, y, ax=axes, color=c8, label='6_7_m')

#axes.set_title("Beam #7 - Moist")
plt.legend(prop={'size': 7})
axes.set_xlabel(x)
axes.set_ylabel(y)
#axes.set_xscale('log')
#axes.get_legend().remove()

plt.savefig('b7_diff.png', dpi=300)
plt.show()
