import pandas as pd

import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.io import wavfile  # get the api
import numpy as np

#Read in files
n11 = pd.read_csv('data/1_1_n.txt', delimiter="\t")
n21 = pd.read_csv('data/2_1_n.txt', delimiter="\t")
n31 = pd.read_csv('data/3_1_n.txt', delimiter="\t")
n41 = pd.read_csv('data/4_1_n.txt', delimiter="\t")
n51 = pd.read_csv('data/5_1_n.txt', delimiter="\t")
n61 = pd.read_csv('data/6_1_n.txt', delimiter="\t")

d1 = pd.read_csv('data/1_4_d.txt', delimiter="\t")
d2 = pd.read_csv('data/2_4_d.txt', delimiter="\t")
d3 = pd.read_csv('data/3_4_d.txt', delimiter="\t")
d4 = pd.read_csv('data/4_4_d.txt', delimiter="\t")
d5 = pd.read_csv('data/5_4_d.txt', delimiter="\t")
d6 = pd.read_csv('data/6_4_d.txt', delimiter="\t")

l1 = pd.read_csv('data/1_7_l.txt', delimiter="\t")
l2 = pd.read_csv('data/2_7_l.txt', delimiter="\t")
l3 = pd.read_csv('data/3_7_l.txt', delimiter="\t")
l4 = pd.read_csv('data/4_7_l.txt', delimiter="\t")
l5 = pd.read_csv('data/5_7_l.txt', delimiter="\t")
l6 = pd.read_csv('data/6_7_l.txt', delimiter="\t")

m1 = pd.read_csv('data/1_7_m.txt', delimiter="\t")
m2 = pd.read_csv('data/2_7_m.txt', delimiter="\t")
m3 = pd.read_csv('data/3_7_m.txt', delimiter="\t")
m4 = pd.read_csv('data/4_7_m.txt', delimiter="\t")
m5 = pd.read_csv('data/5_7_m.txt', delimiter="\t")
m6 = pd.read_csv('data/6_7_m.txt', delimiter="\t")

e1 = pd.read_csv('data/1_4_e.txt', delimiter="\t")
e2 = pd.read_csv('data/2_4_e.txt', delimiter="\t")
e3 = pd.read_csv('data/3_4_e.txt', delimiter="\t")
e4 = pd.read_csv('data/4_4_e.txt', delimiter="\t")
e5 = pd.read_csv('data/5_4_e.txt', delimiter="\t")
e6 = pd.read_csv('data/6_4_e.txt', delimiter="\t")

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

# d1.plot(x, y, ax=axes, color='orange', label="decay")
# d2.plot(x, y, ax=axes[0, 0], color='yellow')
# d3.plot(x, y, ax=axes[0, 0], color='yellow')
# d4.plot(x, y,  ax=axes, color='orange', label="decay")
# d5.plot(x, y, ax=axes[0, 0], color='yellow')
# d6.plot(x, y, ax=axes[0, 0], color='yellow')

# m1.plot(x, y, ax=axes, color='blue', label="moisture")
# # m2.plot(x, y, ax=axes[0, 0], color='pink')
# m3.plot(x, y, ax=axes[0, 0], color='pink')
# m4.plot(x, y, ax=axes, color='lightblue', label="moisture")
# m5.plot(x, y, ax=axes[0, 0], color='pink')
# m6.plot(x, y, ax=axes[0, 0], color='pink')

# l4.plot(x, y, ax=axes, color='lightgreen', label="low moisture")
# l2.plot(x, y, ax=axes[0, 0], color='brown')
# l3.plot(x, y, ax=axes[0, 0], color='brown')
# l4.plot(x, y, ax=axes, color='blue', label="low moisture")
# l5.plot(x, y, ax=axes[0, 0], color='brown')
# l6.plot(x, y, ax=axes[0, 0], color='brown')

# e1.plot(x, y, ax=axes, color='yellow', label="extreme decay")
# e2.plot(x, y, ax=axes[0, 0], color='red')
# e3.plot(x, y, ax=axes[0, 0], color='red')
# e4.plot(x, y, ax=axes, color='darkblue', label="extreme moisture")
# e5.plot(x, y, ax=axes[0, 0], color='red')
# e6.plot(x, y, ax=axes[0, 0], color='red')

axes.set_title("normal")
axes.set_xlabel(x)
axes.set_ylabel(y)
axes.set_xscale('log')
axes.get_legend().remove()

plt.show()
