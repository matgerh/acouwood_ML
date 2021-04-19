import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

#Read in files
n11 = pd.read_csv('../data/1_4_d.txt', delimiter="\t")
n21 = pd.read_csv('../data/2_4_d.txt', delimiter="\t")
n31 = pd.read_csv('../data/3_4_d.txt', delimiter="\t")
n41 = pd.read_csv('../data/4_4_d.txt', delimiter="\t")
n51 = pd.read_csv('../data/5_4_d.txt', delimiter="\t")
n61 = pd.read_csv('../data/6_4_d.txt', delimiter="\t")

fig, axes = plt.subplots()
x = 'Frequency (Hz)'
y = 'Level (dB)'

normal = "#47903A"
decay = "#E3A541"
extremeDecay = "#A2512D"
moisture = "#1878D5"
lightMoisture = "#30A1FB"

# Normal sample 1
n11.plot(x, y, ax=axes, color=decay, label='1_4_d')
n21.plot(x, y, ax=axes, color=decay, label='2_4_d')
n31.plot(x, y, ax=axes, color=decay, label='3_4_d')
n41.plot(x, y, ax=axes, color=extremeDecay, label='4_4_d')
n51.plot(x, y, ax=axes, color=extremeDecay, label='5_4_d')
n61.plot(x, y, ax=axes, color=extremeDecay, label='6_4_d')

axes.set_title("Beam #4 - Decay")
axes.set_xlabel(x)
axes.set_ylabel(y)
#axes.set_xscale('log')
#axes.get_legend().remove()

print('data1: mean=%.3f stdv=%.3f' % (n11[y].mean(), n11[y].std()))
print('data2: mean=%.3f stdv=%.3f' % (n21[y].mean(), n21[y].std()))
print('data3: mean=%.3f stdv=%.3f' % (n31[y].mean(), n31[y].std()))
print('data4: mean=%.3f stdv=%.3f' % (n41[y].mean(), n41[y].std()))
print('data5: mean=%.3f stdv=%.3f' % (n51[y].mean(), n51[y].std()))
print('data6: mean=%.3f stdv=%.3f' % (n61[y].mean(), n61[y].std()))

plt.savefig('w4_decay.png', dpi=300)
plt.show()
