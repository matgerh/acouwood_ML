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
n1 = pd.read_csv('../fft_data/1_1_n.txt', delimiter="\t")
n2 = pd.read_csv('../fft_data/4_1_n.txt', delimiter="\t")
n3 = pd.read_csv('../fft_data/1_2_n.txt', delimiter="\t")
n4 = pd.read_csv('../fft_data/4_2_n.txt', delimiter="\t")
n5 = pd.read_csv('../fft_data/1_3_n.txt', delimiter="\t")
n6 = pd.read_csv('../fft_data/4_3_n.txt', delimiter="\t")
n7 = pd.read_csv('../fft_data/1_4_n.txt', delimiter="\t")
n8 = pd.read_csv('../fft_data/4_4_n.txt', delimiter="\t")
n9 = pd.read_csv('../fft_data/1_5_n.txt', delimiter="\t")
n10 = pd.read_csv('../fft_data/4_5_n.txt', delimiter="\t")
n11 = pd.read_csv('../fft_data/1_6_n.txt', delimiter="\t")
n12 = pd.read_csv('../fft_data/4_6_n.txt', delimiter="\t")

# d1 = pd.read_csv('../fft_data/1_4_d.txt', delimiter="\t")
# d2 = pd.read_csv('../fft_data/4_4_d.txt', delimiter="\t")
# d3 = pd.read_csv('../fft_data/1_5_d.txt', delimiter="\t")
# d4 = pd.read_csv('../fft_data/4_5_d.txt', delimiter="\t")
# d5 = pd.read_csv('../fft_data/1_6_d.txt', delimiter="\t")
# d6 = pd.read_csv('../fft_data/4_6_d.txt', delimiter="\t")

d7 = pd.read_csv('../fft_data/1_4_e.txt', delimiter="\t")
d8 = pd.read_csv('../fft_data/4_4_e.txt', delimiter="\t")
d9 = pd.read_csv('../fft_data/1_5_e.txt', delimiter="\t")
d10 = pd.read_csv('../fft_data/4_5_e.txt', delimiter="\t")
d11 = pd.read_csv('../fft_data/1_6_e.txt', delimiter="\t")
d12 = pd.read_csv('../fft_data/4_6_e.txt', delimiter="\t")

# m1 = pd.read_csv('../fft_data/1_7_m.txt', delimiter="\t")
# m2 = pd.read_csv('../fft_data/4_7_m.txt', delimiter="\t")
# m3 = pd.read_csv('../fft_data/1_8_m.txt', delimiter="\t")
# m4 = pd.read_csv('../fft_data/4_8_m.txt', delimiter="\t")
# m5 = pd.read_csv('../fft_data/1_9_m.txt', delimiter="\t")
# m6 = pd.read_csv('../fft_data/4_9_m.txt', delimiter="\t")

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
y9 = n9[y]
y10 = n10[y]
y100 = n11[y]
y101 = n12[y]
# y11 = d1[y]
# y12 = d2[y]
# y13 = d3[y]
# y14 = d4[y]
# y15 = d5[y]
# y16 = d6[y]
yd7 = d7[y]
yd8 = d8[y]
yd9 = d9[y]
yd10 = d10[y]
yd11 = d11[y]
yd12 = d12[y]
# y17 = m1[y]
# y18 = m2[y]
# y19 = m3[y]
# y20 = m4[y]
# y21 = m5[y]
# y22 = m6[y]

# Normal - wood 1-6
n1.plot(x, y, ax=axes, color=normal, label="normal 1")
n2.plot(x, y, ax=axes, color=normal, label="normal 2")
n3.plot(x, y, ax=axes, color=normal, label="decay 1")
n4.plot(x, y, ax=axes, color=normal, label="decay 2")
n5.plot(x, y, ax=axes, color=normal, label="extreme decay 1")
n6.plot(x, y, ax=axes, color=normal, label="extreme decay 2")
n7.plot(x, y, ax=axes, color=normal, label="moisture 1")
n8.plot(x, y, ax=axes, color=normal, label="moisture 2")
n9.plot(x, y, ax=axes, color=normal, label="light moisture 1")
n10.plot(x, y, ax=axes, color=normal, label="light moisture 2")
n11.plot(x, y, ax=axes, color=normal, label="light moisture 1")
n12.plot(x, y, ax=axes, color=normal, label="light moisture 2")

# Decay wood 4-6
# d1.plot(x, y, ax=axes, color=decay, label="")
# d2.plot(x, y, ax=axes, color=decay, label="")
# d3.plot(x, y, ax=axes, color=decay, label="")
# d4.plot(x, y, ax=axes, color=decay, label="")
# d5.plot(x, y, ax=axes, color=decay, label="")
# d6.plot(x, y, ax=axes, color=decay, label="")


d7.plot(x, y, ax=axes, color=extremeDecay, label="")
d8.plot(x, y, ax=axes, color=extremeDecay, label="")
d9.plot(x, y, ax=axes, color=extremeDecay, label="")
d10.plot(x, y, ax=axes, color=extremeDecay, label="")
d11.plot(x, y, ax=axes, color=extremeDecay, label="")
d12.plot(x, y, ax=axes, color=extremeDecay, label="")

# #Moitsture wood 7-9
# m1.plot(x, y, ax=axes, color=moisture)
# m2.plot(x, y, ax=axes, color=moisture)
# m3.plot(x, y, ax=axes, color=moisture)
# m4.plot(x, y, ax=axes, color=moisture)
# m5.plot(x, y, ax=axes, color=moisture)
# m6.plot(x, y, ax=axes, color=moisture)

axes.set_title(
    "Green = Normal, Brown = Extreme Decay"
)
axes.set_xlabel(x)
axes.set_ylabel(y)
#axes.set_xscale('log')
axes.get_legend().remove()
axes.fill_between(x1, y1, y2, color=normal, interpolate=False, alpha=0.7)
axes.fill_between(x1, y3, y4, color=normal, interpolate=False, alpha=0.7)
axes.fill_between(x1, y5, y6, color=normal, interpolate=False, alpha=0.7)
axes.fill_between(x1, y7, y8, color=normal, interpolate=False, alpha=0.7)
axes.fill_between(x1, y9, y10, color=normal, interpolate=False, alpha=0.7)
axes.fill_between(x1, y100, y101, color=normal, interpolate=False, alpha=0.7)
# axes.fill_between(x1, y11, y12, color=decay, interpolate=False, alpha=0.7)
# axes.fill_between(x1, y13, y14, color=decay, interpolate=False, alpha=0.7)
# axes.fill_between(x1, y15, y16, color=decay, interpolate=False, alpha=0.7)
# axes.fill_between(x1, y15, y16, color=decay, interpolate=False, alpha=0.7)
axes.fill_between(x1, yd7, yd8, color=decay, interpolate=False, alpha=0.7)
axes.fill_between(x1, yd9, yd10, color=decay, interpolate=False, alpha=0.7)
axes.fill_between(x1, yd11, yd12, color=decay, interpolate=False, alpha=0.7)
# axes.fill_between(x1, y17, y18, color=moisture, interpolate=False, alpha=0.7)
# axes.fill_between(x1, y19, y20, color=moisture, interpolate=False, alpha=0.7)
# axes.fill_between(x1, y21, y22, color=moisture, interpolate=False, alpha=0.7)

plt.show()
