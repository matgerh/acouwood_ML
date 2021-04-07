import pandas as pd

import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.io import wavfile  # get the api
import numpy as np

df1 = pd.read_csv('data/1_1_n.txt', delimiter="\t")
df2 = pd.read_csv('data/2_1_n.txt', delimiter="\t")
df3 = pd.read_csv('data/3_1_n.txt', delimiter="\t")
df4 = pd.read_csv('data/4_1_n.txt', delimiter="\t")
df5 = pd.read_csv('data/5_1_n.txt', delimiter="\t")
df6 = pd.read_csv('data/6_1_n.txt', delimiter="\t")

df7 = pd.read_csv('data/2_1_n.txt', delimiter="\t")
df8 = pd.read_csv('data/2_2_n.txt', delimiter="\t")
df9 = pd.read_csv('data/2_3_n.txt', delimiter="\t")
df10 = pd.read_csv('data/2_4_n.txt', delimiter="\t")
df11 = pd.read_csv('data/2_5_n.txt', delimiter="\t")
df12 = pd.read_csv('data/2_6_n.txt', delimiter="\t")

df13 = pd.read_csv('data/3_1_n.txt', delimiter="\t")
df14 = pd.read_csv('data/3_2_n.txt', delimiter="\t")
df15 = pd.read_csv('data/3_3_n.txt', delimiter="\t")
df16 = pd.read_csv('data/3_4_n.txt', delimiter="\t")
df17 = pd.read_csv('data/3_5_n.txt', delimiter="\t")
df18 = pd.read_csv('data/3_6_n.txt', delimiter="\t")

df19 = pd.read_csv('data/4_1_n.txt', delimiter="\t")
df20 = pd.read_csv('data/4_2_n.txt', delimiter="\t")
df21 = pd.read_csv('data/4_3_n.txt', delimiter="\t")
df22 = pd.read_csv('data/4_4_n.txt', delimiter="\t")
df23 = pd.read_csv('data/4_5_n.txt', delimiter="\t")
df24 = pd.read_csv('data/4_6_n.txt', delimiter="\t")


fig, axes = plt.subplots(nrows=2, ncols=3)  # stacking 4 subplots
x = 'Frequency (Hz)'
y = 'Level (dB)'

# midnightblue
df1.plot(x, y, ax=axes[0, 0], color='green')
df2.plot(x, y, ax=axes[0, 0], color='darkgreen')
df3.plot(x, y, ax=axes[0, 0], color='lightgreen')
df4.plot(x, y, ax=axes[0, 0], color='blue')
df5.plot(x, y, ax=axes[0, 0], color='darkblue')
df6.plot(x, y, ax=axes[0, 0], color='lightblue')
axes[0, 0].legend(["1", "2", "3", "4", "5", "6"])
axes[0, 0].set_title("Sample 1 - normal")

df7.plot(x, y, ax=axes[0, 1], color='green')
df8.plot(x, y, ax=axes[0, 1], color='darkgreen')
df9.plot(x, y, ax=axes[0, 1], color='lightgreen')
df10.plot(x, y, ax=axes[0, 1], color='blue')
df11.plot(x, y, ax=axes[0, 1], color='darkblue')
df12.plot(x, y, ax=axes[0, 1], color='lightblue')
axes[0, 1].legend(["1", "2", "3", "4", "5", "6"])
axes[0, 1].set_title("Sample 2 - normal")

df13.plot(x, y, ax=axes[1, 0], color='green')
df14.plot(x, y, ax=axes[1, 0], color='darkgreen')
df15.plot(x, y, ax=axes[1, 0], color='lightgreen')
df16.plot(x, y, ax=axes[1, 0], color='blue')
df17.plot(x, y, ax=axes[1, 0], color='darkblue')
df18.plot(x, y, ax=axes[1, 0], color='lightblue')
axes[1, 0].legend(["1", "2", "3", "4", "5", "6"])
axes[1, 0].set_title("Sample 3 - normal")

df19.plot(x, y, ax=axes[1, 1], color='green')
df20.plot(x, y, ax=axes[1, 1], color='darkgreen')
df21.plot(x, y, ax=axes[1, 1], color='lightgreen')
df22.plot(x, y, ax=axes[1, 1], color='blue')
df23.plot(x, y, ax=axes[1, 1], color='darkblue')
df24.plot(x, y, ax=axes[1, 1], color='lightblue')
axes[1, 1].legend(["1", "2", "3", "4", "5", "6"])
axes[1, 1].set_title("Sample 4 - normal")

plt.show()
