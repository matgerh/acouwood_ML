import os
import pandas as pd
from math import sqrt
import mlflow
import matplotlib.pyplot as plt
from statistics import mean

pd.set_option('display.max_rows', None)


data = []

#Read in files
n11 = pd.read_csv('fft_data/1_1_n.txt', delimiter="\t")
n41 = pd.read_csv('fft_data/4_1_n.txt', delimiter="\t")
n12 = pd.read_csv('fft_data/1_2_n.txt', delimiter="\t")
n42 = pd.read_csv('fft_data/4_2_n.txt', delimiter="\t")
n13 = pd.read_csv('fft_data/1_3_n.txt', delimiter="\t")
n43 = pd.read_csv('fft_data/4_3_n.txt', delimiter="\t")

n14 = pd.read_csv('fft_data/1_4_n.txt', delimiter="\t")
n44 = pd.read_csv('fft_data/4_4_n.txt', delimiter="\t")
n15 = pd.read_csv('fft_data/1_5_n.txt', delimiter="\t")
n45 = pd.read_csv('fft_data/4_5_n.txt', delimiter="\t")
n16 = pd.read_csv('fft_data/1_6_n.txt', delimiter="\t")
n46 = pd.read_csv('fft_data/4_6_n.txt', delimiter="\t")

d14 = pd.read_csv('fft_data/1_4_d.txt', delimiter="\t")
d24 = pd.read_csv('fft_data/2_4_d.txt', delimiter="\t")
d34 = pd.read_csv('fft_data/3_4_d.txt', delimiter="\t")
d44 = pd.read_csv('fft_data/4_4_d.txt', delimiter="\t")
d54 = pd.read_csv('fft_data/5_4_d.txt', delimiter="\t")
d64 = pd.read_csv('fft_data/6_4_d.txt', delimiter="\t")


d44 = pd.read_csv('fft_data/4_4_d.txt', delimiter="\t")
d15 = pd.read_csv('fft_data/1_5_d.txt', delimiter="\t")
d45 = pd.read_csv('fft_data/4_5_d.txt', delimiter="\t")
d16 = pd.read_csv('fft_data/1_6_d.txt', delimiter="\t")
d46 = pd.read_csv('fft_data/4_6_d.txt', delimiter="\t")

n17 = pd.read_csv('fft_data/1_7_n.txt', delimiter="\t")
n27 = pd.read_csv('fft_data/2_7_n.txt', delimiter="\t")
n37 = pd.read_csv('fft_data/3_7_n.txt', delimiter="\t")
n47 = pd.read_csv('fft_data/4_7_n.txt', delimiter="\t")
n57 = pd.read_csv('fft_data/4_7_n.txt', delimiter="\t")
n67 = pd.read_csv('fft_data/4_7_n.txt', delimiter="\t")

n47 = pd.read_csv('fft_data/4_7_n.txt', delimiter="\t")
n18 = pd.read_csv('fft_data/1_8_n.txt', delimiter="\t")
n48 = pd.read_csv('fft_data/4_8_n.txt', delimiter="\t")
n19 = pd.read_csv('fft_data/1_9_n.txt', delimiter="\t")
n49 = pd.read_csv('fft_data/4_9_n.txt', delimiter="\t")

m17 = pd.read_csv('fft_data/1_7_m.txt', delimiter="\t")
m27 = pd.read_csv('fft_data/2_7_m.txt', delimiter="\t")
m37 = pd.read_csv('fft_data/3_7_m.txt', delimiter="\t")
m47 = pd.read_csv('fft_data/4_7_m.txt', delimiter="\t")
m57 = pd.read_csv('fft_data/5_7_m.txt', delimiter="\t")
m67 = pd.read_csv('fft_data/6_7_m.txt', delimiter="\t")

m18 = pd.read_csv('fft_data/1_8_m.txt', delimiter="\t")
m48 = pd.read_csv('fft_data/4_8_m.txt', delimiter="\t")
m19 = pd.read_csv('fft_data/1_9_m.txt', delimiter="\t")
m49 = pd.read_csv('fft_data/4_9_m.txt', delimiter="\t")

e14 = pd.read_csv('fft_data/1_4_e.txt', delimiter="\t")
e24 = pd.read_csv('fft_data/2_4_e.txt', delimiter="\t")
e34 = pd.read_csv('fft_data/3_4_e.txt', delimiter="\t")
e44 = pd.read_csv('fft_data/4_4_e.txt', delimiter="\t")
e54 = pd.read_csv('fft_data/5_4_e.txt', delimiter="\t")
e64 = pd.read_csv('fft_data/6_4_e.txt', delimiter="\t")

e15 = pd.read_csv('fft_data/1_5_e.txt', delimiter="\t")
e45 = pd.read_csv('fft_data/4_5_e.txt', delimiter="\t")
e16 = pd.read_csv('fft_data/1_6_e.txt', delimiter="\t")
e46 = pd.read_csv('fft_data/4_6_e.txt', delimiter="\t")

n21 = pd.read_csv('fft_data/2_1_n.txt', delimiter="\t")
n31 = pd.read_csv('fft_data/3_1_n.txt', delimiter="\t")
n51 = pd.read_csv('fft_data/5_1_n.txt', delimiter="\t")
n61 = pd.read_csv('fft_data/6_1_n.txt', delimiter="\t")

fig, axes = plt.subplots()
x = 'Frequency (Hz)'
y = 'Level (dB)'


# n and d - different samples
c01 = n11[y].corr(d14[y])
c02 = n41[y].corr(d44[y])
c1 = n12[y].corr(d15[y])
c2 = n42[y].corr(d45[y])
c4 = n13[y].corr(d16[y])
c5 = n43[y].corr(d46[y])

c6 = n14[y].corr(d14[y])
c7 = n44[y].corr(d44[y])
c8 = n15[y].corr(d15[y])
c9 = n45[y].corr(d45[y])
c10 = n16[y].corr(d16[y])
c11 = n46[y].corr(d46[y])

c12 = n17[y].corr(d14[y])
c13 = n47[y].corr(d44[y])
c14 = n18[y].corr(d15[y])
c15 = n48[y].corr(d45[y])
c015 = n19[y].corr(d16[y])
c016 = n49[y].corr(d46[y])

# n and e - different samples
c31 = n11[y].corr(e14[y])
c32 = n41[y].corr(e44[y])
c33 = n12[y].corr(e15[y])
c34 = n42[y].corr(e45[y])
c35 = n13[y].corr(e16[y])
c36 = n43[y].corr(e46[y])

c37 = n14[y].corr(e15[y])
c38 = n44[y].corr(e45[y])
c39 = n15[y].corr(e16[y])
c40 = n45[y].corr(e46[y])
c41 = n16[y].corr(e15[y])
c42 = n46[y].corr(e45[y])

c43 = n17[y].corr(e14[y])
c44 = n47[y].corr(e44[y])
c45 = n18[y].corr(e15[y])
c46 = n48[y].corr(e45[y])
c47 = n19[y].corr(e16[y])
c48 = n49[y].corr(e46[y])


# n and m different samples
c666 = n11[y].corr(m17[y])
c777 = n41[y].corr(m47[y])
c888 = n12[y].corr(m18[y])
c16 = n42[y].corr(m48[y])
c17 = n13[y].corr(m19[y])
c18 = n43[y].corr(m49[y])

c19 = n14[y].corr(m17[y])
c20 = n44[y].corr(m47[y])
c21 = n15[y].corr(m18[y])
c22 = n45[y].corr(m48[y])
c23 = n16[y].corr(m19[y])
c24 = n46[y].corr(m49[y])

c25 = n17[y].corr(m18[y])
c26 = n47[y].corr(m48[y])
c27 = n18[y].corr(m19[y])
c28 = n48[y].corr(m49[y])
c29 = n19[y].corr(m17[y])
c30 = n49[y].corr(m47[y])

# e and m
c68 = e14[y].corr(m17[y])
c69 = e44[y].corr(m47[y])
c70 = e15[y].corr(m18[y])
c71 = e45[y].corr(m48[y])
c72 = e16[y].corr(m19[y])
c73 = e46[y].corr(m49[y])

# d og e - different samples
c86 = e14[y].corr(d15[y])
c87 = e44[y].corr(d45[y])
c88 = e15[y].corr(d16[y])
c89 = e45[y].corr(d46[y])
c90 = e16[y].corr(d14[y])
c91 = e46[y].corr(d44[y])

# n - same sample & mounting 
c50 = n11[y].corr(n21[y])
c51 = n21[y].corr(n31[y])
c52 = n11[y].corr(n31[y])

# e - same sample & mounting
c53 = e14[y].corr(e24[y])
c54 = e24[y].corr(e34[y])
c55 = e14[y].corr(e34[y])

# m - same sample & mounting
c92 = m17[y].corr(m27[y])
c93 = m27[y].corr(m37[y])
c94 = m37[y].corr(m17[y])

# n same sample - different mounting
c56 = n11[y].corr(n41[y])
c57 = n21[y].corr(n51[y])
c58 = n31[y].corr(n61[y])

c59 = n17[y].corr(n47[y])
c60 = n27[y].corr(n57[y])
c61 = n37[y].corr(n67[y])

# e same sample - different mounting
c62 = e14[y].corr(e44[y])
c63 = e24[y].corr(e54[y])
c64 = e34[y].corr(e64[y])

# m same sample - different mounting
c65 = m17[y].corr(m47[y])
c66 = m27[y].corr(m57[y])
c67 = m37[y].corr(m67[y])

# n and e - same sample nr
c74 = n14[y].corr(e14[y])
c75 = n44[y].corr(e44[y])
c76 = n15[y].corr(e15[y])
c77 = n45[y].corr(e45[y])
c78 = n16[y].corr(e16[y])
c79 = n46[y].corr(e46[y])

# n and m - same sam number
c80 = n17[y].corr(m17[y])
c81 = n47[y].corr(m47[y])
c82 = n18[y].corr(m18[y])
c83 = n48[y].corr(m48[y])
c84 = n19[y].corr(m19[y])
c85 = n49[y].corr(m19[y])

# n - different sample number
c95 = n11[y].corr(n12[y])
c96 = n13[y].corr(n14[y])
c97 = n14[y].corr(n15[y])

c98 = n17[y].corr(n18[y])
c99 = n19[y].corr(n41[y])
c100 = n43[y].corr(n67[y])

# m different sample
c101 = m17[y].corr(m48[y])
c102 = m47[y].corr(m49[y])
c103 = m19[y].corr(m18[y])

# e different sample
c104 = e14[y].corr(e15[y])
c105 = e16[y].corr(e44[y])
c106 = e45[y].corr(e46[y])


# d og m - different sample
c107 = m17[y].corr(d15[y])
c108 = m47[y].corr(d45[y])
c109 = m18[y].corr(d16[y])
c110 = m48[y].corr(d46[y])
c111 = m19[y].corr(d14[y])
c112 = m49[y].corr(d44[y])


# d - same sample & mounting
c113 = d14[y].corr(d24[y])
c114 = d24[y].corr(d34[y])
c115 = d14[y].corr(d34[y])

# d - same sample & mounting
c116 = d14[y].corr(d44[y])
c117 = d24[y].corr(d54[y])
c118 = d34[y].corr(d64[y])

# d different sample
c119 = d14[y].corr(d15[y])
c120 = d16[y].corr(d44[y])
c121 = d45[y].corr(d46[y])

# n og d - same sample
c122 = d14[y].corr(n14[y])
c123 = d15[y].corr(n15[y])
c124 = d16[y].corr(n16[y])

# d and e - same number
c125 = d14[y].corr(e14[y])
c126 = d44[y].corr(e44[y])
c127 = d15[y].corr(e15[y])
c128 = d45[y].corr(e45[y])
c129 = d16[y].corr(e16[y])
c130 = d46[y].corr(e46[y])





print("e same sample and mounting:")
print(mean([c53,c54,c55]))
print("n same sample and mounting:")
print(mean([c50,c51,c52]))
print("m same sample and mounting:")
print(mean([c92,c93,c94]))

print("   ")

print("n og d different samples:")
print(mean([c01,c02,c2,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c015,c016]))
print("n og e different samples:") 
print(mean([c31,c32,c33,c34,c35,c36,c37,c38,c39,c40,c41,c42,c43,c44,c45,c46,c47,c48]))
print("n og m different samples:")
print(mean([c666,c777,c888,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,c26,c27,c28,c29,c30]))
print("e og m different samples:")
print(mean([c68,c69,c70,c71,c72,c73]))
print("e og n - same samp number")
print(mean([c74,c75,c76,c77,c78,c79]))
print("n og m - same samp number")
print(mean([c80,c81,c82,c83,c84,c85]))
print("  ")
print("e og d - different number:")
print(mean([c86,c87,c88,c89,c90,c91]))

print("n same sample different mounting:")
print(mean([c56,c57,c58,c59,c60,c61]))
print("e same sample different mounting:")
print(mean([c62,c63,c64]))
print("m same sample different mounting:")
print(mean([c65,c66,c67]))


print("n different samples")
print(mean([c95,c96,c97,c98,c99,c100]))

print("m different samples")
print(mean([c101,c102,c103]))

print("e different samples")
print(mean([c104,c105,c106]))

print("d og m - differnt samples")
print(mean([c107,c108,c109,c110,c111,c112]))

print("d same sample and mounting")
print(mean([c113,c114,c115]))

print("d same sample different mounting")
print(mean([c116,c117,c118]))

print("d different samples")
print(mean([c119,c120,c121]))

print("n og d -same sample")
print(mean([c122,c123,c124]))

print("d og e - same number")
print(mean([c125,c126,c127,c128,c129,c130]))












