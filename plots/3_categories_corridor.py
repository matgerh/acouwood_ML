import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

#colors
normal = "#47903A"
decay = "#E3A541"
extremeDecay = "#A2512D"
moisture = "#1878D5"
lightMoisture = "#30A1FB"

#Read in files
n1 = pd.read_csv('../data/1_1_n.txt', delimiter="\t")
n2 = pd.read_csv('../data/1_7_l.txt', delimiter="\t")
n3 = pd.read_csv('../data/1_4_d.txt', delimiter="\t")
n4 = pd.read_csv('../data/4_4_d.txt', delimiter="\t")
n5 = pd.read_csv('../data/1_8_m.txt', delimiter="\t")
n6 = pd.read_csv('../data/1_9_m.txt', delimiter="\t")
n7 = pd.read_csv('../data/1_5_e.txt', delimiter="\t")
n8 = pd.read_csv('../data/4_5_e.txt', delimiter="\t")

fig, axes = plt.subplots()
x = 'Frequency (Hz)'
y = 'Level (dB)'

x1 = n1[x]
y1 = n1[y]
y2 = n2[y]
y3 = n3[y]
y4 = n4[y]
y5 = n5[y]
y6 = n6[y]
y7 = n7[y]
y8 = n8[y]

# Normal sample 1
n1.plot(x, y, ax=axes, color=normal, label="normal (1_1_n")
n2.plot(x, y, ax=axes, color=normal, label="normal (1_7_l)")
n3.plot(x, y, ax=axes, color=decay, label="decay (1_4_d)")
n4.plot(x, y, ax=axes, color=decay, label="decay (4_4_d)")
n5.plot(x, y, ax=axes, color=moisture, label="moisture (1_8_m)")
n6.plot(x, y, ax=axes, color=moisture, label="moisture (1_9_m)")
n7.plot(x, y, ax=axes, color=extremeDecay, label="extreme decay (1_5_e)")
n8.plot(x, y, ax=axes, color=extremeDecay, label="extreme decay (4_5_e)")

axes.set_title(
    "Overview of the four decided categories - the width of the curve indicates the difference between mountings."
)
axes.set_xlabel(x)
axes.set_ylabel(y)
#axes.set_xscale('log')
#axes.get_legend().remove()
axes.fill_between(x1, y1, y2, color=normal, interpolate=True, alpha=0.9)
axes.fill_between(x1, y3, y4, color=decay, interpolate=True, alpha=0.8)
axes.fill_between(x1, y5, y6, color=moisture, interpolate=True, alpha=0.7)
axes.fill_between(x1, y7, y8, color=extremeDecay, interpolate=True, alpha=0.6)

plt.show()
