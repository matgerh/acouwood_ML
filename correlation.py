import os
import pandas as pd
from math import sqrt
import mlflow
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)


data = []

#Read in files
n11 = pd.read_csv('data/1_1_n.txt', delimiter="\t")
n41 = pd.read_csv('data/4_1_n.txt', delimiter="\t")
n12 = pd.read_csv('data/1_2_n.txt', delimiter="\t")
n42 = pd.read_csv('data/4_2_n.txt', delimiter="\t")
n13 = pd.read_csv('data/1_3_n.txt', delimiter="\t")
n43 = pd.read_csv('data/4_3_n.txt', delimiter="\t")

d14 = pd.read_csv('data/1_4_d.txt', delimiter="\t")
d44 = pd.read_csv('data/4_4_d.txt', delimiter="\t")
d15 = pd.read_csv('data/1_5_d.txt', delimiter="\t")
d45 = pd.read_csv('data/4_5_d.txt', delimiter="\t")
d16 = pd.read_csv('data/1_6_d.txt', delimiter="\t")
d46 = pd.read_csv('data/4_6_d.txt', delimiter="\t")

l17 = pd.read_csv('data/1_7_l.txt', delimiter="\t")
l47 = pd.read_csv('data/4_7_l.txt', delimiter="\t")
l18 = pd.read_csv('data/1_8_l.txt', delimiter="\t")
l48 = pd.read_csv('data/4_8_l.txt', delimiter="\t")
l19 = pd.read_csv('data/1_9_l.txt', delimiter="\t")
l49 = pd.read_csv('data/4_9_l.txt', delimiter="\t")

m17 = pd.read_csv('data/1_7_m.txt', delimiter="\t")
m47 = pd.read_csv('data/4_7_m.txt', delimiter="\t")
m18 = pd.read_csv('data/1_8_m.txt', delimiter="\t")
m48 = pd.read_csv('data/4_8_m.txt', delimiter="\t")
m19 = pd.read_csv('data/1_9_m.txt', delimiter="\t")
m49 = pd.read_csv('data/4_9_m.txt', delimiter="\t")

e14 = pd.read_csv('data/1_4_e.txt', delimiter="\t")
e44 = pd.read_csv('data/4_4_e.txt', delimiter="\t")
e15 = pd.read_csv('data/1_5_e.txt', delimiter="\t")
e45 = pd.read_csv('data/4_5_e.txt', delimiter="\t")
e16 = pd.read_csv('data/1_6_e.txt', delimiter="\t")
e46 = pd.read_csv('data/4_6_e.txt', delimiter="\t")

n21 = pd.read_csv('data/2_1_n.txt', delimiter="\t")
n31 = pd.read_csv('data/3_1_n.txt', delimiter="\t")
n51 = pd.read_csv('data/5_1_n.txt', delimiter="\t")
n61 = pd.read_csv('data/6_1_n.txt', delimiter="\t")

fig, axes = plt.subplots()
x = 'Frequency (Hz)'
y = 'Level (dB)'

c1 = n11[y].corr(d14[y])
c2 = n41[y].corr(d44[y])
c3 = n12[y].corr(d15[y])
c4 = n42[y].corr(d45[y])
c5 = n13[y].corr(d16[y])
c6 = n43[y].corr(d46[y])

c7 = n11[y].corr(n21[y])
c8 = n21[y].corr(n31[y])
c9 = n11[y].corr(n31[y])


c10 = n11[y].corr(e14[y])
c11 = n41[y].corr(e44[y])
c12 = n12[y].corr(e15[y])

c13 = n11[y].corr(n41[y])
c14 = n21[y].corr(n51[y])
c15 = n31[y].corr(n61[y])

c16 = n11[y].corr(m17[y])
c17 = n41[y].corr(m47[y])
c18 = n12[y].corr(m18[y])


c16 = n11[y].corr(l17[y])
c17 = n41[y].corr(l47[y])
c18 = n12[y].corr(l18[y])

print(c1,c2,c3,c4,c5,c6,c7,c8,c9)
print(c10,c11,c12)
print(c13,c14,c15)
print(c16,c17,c18)
exit()












