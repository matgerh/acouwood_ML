import pandas as pd

import matplotlib.pyplot as plt
from scipy.fftpack import fft,fftfreq
from scipy.io import wavfile # get the api
import numpy as np

df1 = pd.read_csv('data/1_4_n.txt', delimiter = "\t")
df2 = pd.read_csv('data/1_4_e.txt', delimiter = "\t")

df3 = pd.read_csv('data/1_5_n.txt', delimiter = "\t")
df4 = pd.read_csv('data/1_5_e.txt', delimiter = "\t")

df5 = pd.read_csv('data/1_6_n.txt', delimiter = "\t")
df6 = pd.read_csv('data/1_6_e.txt', delimiter = "\t")

df7 = pd.read_csv('data/3_6_n.txt', delimiter = "\t")
df8 = pd.read_csv('data/3_6_e.txt', delimiter = "\t")


fig, axes = plt.subplots(nrows=2, ncols=2) # stacking 4 subplots 
x = 'Frequency (Hz)'
y = 'Level (dB)'


df1.plot(x, y, ax=axes[0,0], color='grey')
df2.plot(x, y, ax=axes[0,0], color='springgreen')
axes[0,0].legend(["4_n","4_e"])

df3.plot(x, y, ax=axes[0,1], color='grey')
df4.plot(x, y, ax=axes[0,1], color='springgreen')
axes[0,1].legend(["5_n","5_e"])

df5.plot(x, y, ax=axes[1,0], color='grey')
df6.plot(x, y, ax=axes[1,0], color='springgreen')
axes[1,0].legend(["6_n","6_e"])

df7.plot(x, y, ax=axes[1,1], color='grey')
df8.plot(x, y, ax=axes[1,1], color='springgreen')
axes[1,1].legend(["6_n","6_e"])

plt.show()