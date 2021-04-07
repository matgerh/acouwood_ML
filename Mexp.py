import pandas as pd

import matplotlib.pyplot as plt
from scipy.fftpack import fft,fftfreq
from scipy.io import wavfile # get the api
import numpy as np
from sklearn.metrics import r2_score
######################################

#Calculating r2
df1 = pd.read_csv('data/1_1_n.txt', delimiter = "\t")
df2 = pd.read_csv('data/2_1_n.txt', delimiter = "\t")
first = r2_score(df1['Level (dB)'],df2['Level (dB)'])

df1 = pd.read_csv('data/1_1_n.txt', delimiter = "\t")
df2 = pd.read_csv('data/4_1_n.txt', delimiter = "\t")
second = r2_score(df1['Level (dB)'],df2['Level (dB)'])

df1 = pd.read_csv('data/1_1_n.txt', delimiter = "\t")
df2 = pd.read_csv('data/3_8_m.txt', delimiter = "\t")
third = r2_score(df1['Level (dB)'],df2['Level (dB)'])

print(first)
print(second)
print(third)
exit()



############################################




df1 = pd.read_csv('data/1_1_n.txt', delimiter = "\t")
df2 = pd.read_csv('data/1_7_m.txt', delimiter = "\t")

df3 = pd.read_csv('data/1_7_l.txt', delimiter = "\t")
df4 = pd.read_csv('data/1_7_m.txt', delimiter = "\t")

df5 = pd.read_csv('data/1_9_l.txt', delimiter = "\t")
df6 = pd.read_csv('data/1_9_m.txt', delimiter = "\t")

df7 = pd.read_csv('data/3_7_l.txt', delimiter = "\t")
df8 = pd.read_csv('data/3_7_m.txt', delimiter = "\t")


fig, axes = plt.subplots(nrows=2, ncols=2) # stacking 4 subplots 
x = 'Frequency (Hz)'
y = 'Level (dB)'

# midnightblue
df1.plot(x, y, ax=axes[0,0], color='darkred')
df2.plot(x, y, ax=axes[0,0], color='darkkhaki')
axes[0,0].legend(["1_n","2_n"])

df3.plot(x, y, ax=axes[0,1], color='darkred')
df4.plot(x, y, ax=axes[0,1], color='darkkhaki')
axes[0,1].legend(["3_n","5_m"])

df5.plot(x, y, ax=axes[1,0], color='darkred')
df6.plot(x, y, ax=axes[1,0], color='darkkhaki')
axes[1,0].legend(["6_l","6_m"])

df7.plot(x, y, ax=axes[1,1], color='darkred')
df8.plot(x, y, ax=axes[1,1], color='darkkhaki')
axes[1,1].legend(["6_l","6_m"])

plt.show()