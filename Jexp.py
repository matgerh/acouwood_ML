import pandas as pd

import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.io import wavfile  # get the api
import numpy as np

n1 = pd.read_csv('data/1_1_n.txt', delimiter="\t")
n2 = pd.read_csv('data/2_1_n.txt', delimiter="\t")
n3 = pd.read_csv('data/3_1_n.txt', delimiter="\t")
n4 = pd.read_csv('data/4_1_n.txt', delimiter="\t")
n5 = pd.read_csv('data/5_1_n.txt', delimiter="\t")
n6 = pd.read_csv('data/6_1_n.txt', delimiter="\t")

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


fig, axes = plt.subplots()  # stacking 4 subplots
x = 'Frequency (Hz)'
y = 'Level (dB)'

# midnightblue
n1.plot(x, y, ax=axes, color='darkgreen', label="normal")
# n2.plot(x, y, ax=axes[0, 0], color='green')
# n3.plot(x, y, ax=axes[0, 0], color='green')
# n4.plot(x, y, ax=axes, color='darkgreen', label="normal")
# n5.plot(x, y, ax=axes[0, 0], color='green')
# n6.plot(x, y, ax=axes[0, 0], color='green')

d1.plot(x, y, ax=axes, color='orange', label="decay")
# d2.plot(x, y, ax=axes[0, 0], color='yellow')
# d3.plot(x, y, ax=axes[0, 0], color='yellow')
# d4.plot(x, y,  ax=axes, color='orange', label="decay")
# d5.plot(x, y, ax=axes[0, 0], color='yellow')
# d6.plot(x, y, ax=axes[0, 0], color='yellow')

m1.plot(x, y, ax=axes, color='lightblue', label="moisture")
# m2.plot(x, y, ax=axes[0, 0], color='pink')
# m3.plot(x, y, ax=axes[0, 0], color='pink')
# m4.plot(x, y, ax=axes, color='lightblue', label="moisture")
# m5.plot(x, y, ax=axes[0, 0], color='pink')
# m6.plot(x, y, ax=axes[0, 0], color='pink')

l1.plot(x, y, ax=axes, color='blue', label="low moisture")
# l2.plot(x, y, ax=axes[0, 0], color='brown')
# l3.plot(x, y, ax=axes[0, 0], color='brown')
# l4.plot(x, y, ax=axes, color='blue', label="low moisture")
# l5.plot(x, y, ax=axes[0, 0], color='brown')
# l6.plot(x, y, ax=axes[0, 0], color='brown')

e1.plot(x, y, ax=axes, color='darkblue', label="extreme moisture")
# e2.plot(x, y, ax=axes[0, 0], color='red')
# e3.plot(x, y, ax=axes[0, 0], color='red')
# e4.plot(x, y, ax=axes, color='darkblue', label="extreme moisture")
# e5.plot(x, y, ax=axes[0, 0], color='red')
# e6.plot(x, y, ax=axes[0, 0], color='red')


axes.set_title("Comparison of categories across samples")
axes.set_xlabel(x)
axes.set_ylabel(y)


plt.show()
