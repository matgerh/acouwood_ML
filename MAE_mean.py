import os
import pandas as pd
from math import sqrt
import mlflow
import matplotlib.pyplot as plt
from statistics import mean

pd.set_option('display.max_rows', None)


data = []

#Read in files
n11 = pd.read_csv('fft_data/1_1_c.txt', delimiter="\t")
n41 = pd.read_csv('fft_data/4_1_c.txt', delimiter="\t")
n12 = pd.read_csv('fft_data/1_2_c.txt', delimiter="\t")
n42 = pd.read_csv('fft_data/4_2_c.txt', delimiter="\t")
n13 = pd.read_csv('fft_data/1_3_c.txt', delimiter="\t")
n43 = pd.read_csv('fft_data/4_3_c.txt', delimiter="\t")

n14 = pd.read_csv('fft_data/1_4_c.txt', delimiter="\t")
n44 = pd.read_csv('fft_data/4_4_c.txt', delimiter="\t")
n15 = pd.read_csv('fft_data/1_5_c.txt', delimiter="\t")
n45 = pd.read_csv('fft_data/4_5_c.txt', delimiter="\t")
n16 = pd.read_csv('fft_data/1_6_c.txt', delimiter="\t")
n46 = pd.read_csv('fft_data/4_6_c.txt', delimiter="\t")

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

n17 = pd.read_csv('fft_data/1_7_c.txt', delimiter="\t")
n27 = pd.read_csv('fft_data/2_7_c.txt', delimiter="\t")
n37 = pd.read_csv('fft_data/3_7_c.txt', delimiter="\t")
n47 = pd.read_csv('fft_data/4_7_c.txt', delimiter="\t")
n57 = pd.read_csv('fft_data/4_7_c.txt', delimiter="\t")
n67 = pd.read_csv('fft_data/4_7_c.txt', delimiter="\t")

n47 = pd.read_csv('fft_data/4_7_c.txt', delimiter="\t")
n18 = pd.read_csv('fft_data/1_8_c.txt', delimiter="\t")
n48 = pd.read_csv('fft_data/4_8_c.txt', delimiter="\t")
n19 = pd.read_csv('fft_data/1_9_c.txt', delimiter="\t")
n49 = pd.read_csv('fft_data/4_9_c.txt', delimiter="\t")

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

n21 = pd.read_csv('fft_data/2_1_c.txt', delimiter="\t")
n31 = pd.read_csv('fft_data/3_1_c.txt', delimiter="\t")
n51 = pd.read_csv('fft_data/5_1_c.txt', delimiter="\t")
n61 = pd.read_csv('fft_data/6_1_c.txt', delimiter="\t")

fig, axes = plt.subplots()
x = 'Frequency (Hz)'
y = 'Level (dB)'



from sklearn.metrics import mean_absolute_error as MAE


# n and d - different samples
c01 = MAE(n11[y],d14[y])
c02 = MAE(n41[y],d44[y])
c1 = MAE(n12[y],d15[y])
c2 = MAE(n42[y],d45[y])
c4 = MAE(n13[y],d16[y])
c5 = MAE(n43[y],d46[y])

c6 = MAE(n14[y],d14[y])
c7 = MAE(n44[y],d44[y])
c8 = MAE(n15[y],d15[y])
c9 = MAE(n45[y],d45[y])
c10 = MAE(n16[y],d16[y])
c11 = MAE(n46[y],d46[y])

c12 = MAE(n17[y],d14[y])
c13 = MAE(n47[y],d44[y])
c14 = MAE(n18[y],d15[y])
c15 = MAE(n48[y],d45[y])
c015 = MAE(n19[y],d16[y])
c016 = MAE(n49[y],d46[y])

# n and e - different samples
c31 = MAE(n11[y],e14[y])
c32 = MAE(n41[y],e44[y])
c33 = MAE(n12[y],e15[y])
c34 = MAE(n42[y],e45[y])
c35 = MAE(n13[y],e16[y])
c36 = MAE(n43[y],e46[y])

c37 = MAE(n14[y],e15[y])
c38 = MAE(n44[y],e45[y])
c39 = MAE(n15[y],e16[y])
c40 = MAE(n45[y],e46[y])
c41 = MAE(n16[y],e15[y])
c42 = MAE(n46[y],e45[y])

c43 = MAE(n17[y],e14[y])
c44 = MAE(n47[y],e44[y])
c45 = MAE(n18[y],e15[y])
c46 = MAE(n48[y],e45[y])
c47 = MAE(n19[y],e16[y])
c48 = MAE(n49[y],e46[y])


# n and m different samples
c666 = MAE(n11[y],m17[y])
c777 = MAE(n41[y],m47[y])
c888 = MAE(n12[y],m18[y])
c16 = MAE(n42[y],m48[y])
c17 = MAE(n13[y],m19[y])
c18 = MAE(n43[y],m49[y])

c19 = MAE(n14[y],m17[y])
c20 = MAE(n44[y],m47[y])
c21 = MAE(n15[y],m18[y])
c22 = MAE(n45[y],m48[y])
c23 = MAE(n16[y],m19[y])
c24 = MAE(n46[y],m49[y])

c25 = MAE(n17[y],m18[y])
c26 = MAE(n47[y],m48[y])
c27 = MAE(n18[y],m19[y])
c28 = MAE(n48[y],m49[y])
c29 = MAE(n19[y],m17[y])
c30 = MAE(n49[y],m47[y])

# e and m
c68 = MAE(e14[y],m17[y])
c69 = MAE(e44[y],m47[y])
c70 = MAE(e15[y],m18[y])
c71 = MAE(e45[y],m48[y])
c72 = MAE(e16[y],m19[y])
c73 = MAE(e46[y],m49[y])

# d og e - different samples
c86 = MAE(e14[y],d15[y])
c87 = MAE(e44[y],d45[y])
c88 = MAE(e15[y],d16[y])
c89 = MAE(e45[y],d46[y])
c90 = MAE(e16[y],d14[y])
c91 = MAE(e46[y],d44[y])

# n - same sample & mounting 
c50 = MAE(n11[y],n21[y])
c51 = MAE(n21[y],n31[y])
c52 = MAE(n11[y],n31[y])

# e - same sample & mounting
c53 = MAE(e14[y],e24[y])
c54 = MAE(e24[y],e34[y])
c55 = MAE(e14[y],e34[y])

# m - same sample & mounting
c92 = MAE(m17[y],m27[y])
c93 = MAE(m27[y],m37[y])
c94 = MAE(m37[y],m17[y])

# n same sample - different mounting
c56 = MAE(n11[y],n41[y])
c57 = MAE(n21[y],n51[y])
c58 = MAE(n31[y],n61[y])

c59 = MAE(n17[y],n47[y])
c60 = MAE(n27[y],n57[y])
c61 = MAE(n37[y],n67[y])

# e same sample - different mounting
c62 = MAE(e14[y],e44[y])
c63 = MAE(e24[y],e54[y])
c64 = MAE(e34[y],e64[y])

# m same sample - different mounting
c65 = MAE(m17[y],m47[y])
c66 = MAE(m27[y],m57[y])
c67 = MAE(m37[y],m67[y])

# n and e - same sample nr
c74 = MAE(n14[y],e14[y])
c75 = MAE(n44[y],e44[y])
c76 = MAE(n15[y],e15[y])
c77 = MAE(n45[y],e45[y])
c78 = MAE(n16[y],e16[y])
c79 = MAE(n46[y],e46[y])

# n and m - same sam number
c80 = MAE(n17[y],m17[y])
c81 = MAE(n47[y],m47[y])
c82 = MAE(n18[y],m18[y])
c83 = MAE(n48[y],m48[y])
c84 = MAE(n19[y],m19[y])
c85 = MAE(n49[y],m19[y])

# n - different sample number
c95 = MAE(n11[y],n12[y])
c96 = MAE(n13[y],n14[y])
c97 = MAE(n14[y],n15[y])

c98 = MAE(n17[y],n18[y])
c99 = MAE(n19[y],n41[y])
c100 = MAE(n43[y],n67[y])

# m different sample
c101 = MAE(m17[y],m48[y])
c102 = MAE(m47[y],m49[y])
c103 = MAE(m19[y],m18[y])

# e different sample
c104 = MAE(e14[y],e15[y])
c105 = MAE(e16[y],e44[y])
c106 = MAE(e45[y],e46[y])


# d og m - different sample
c107 = MAE(m17[y],d15[y])
c108 = MAE(m47[y],d45[y])
c109 = MAE(m18[y],d16[y])
c110 = MAE(m48[y],d46[y])
c111 = MAE(m19[y],d14[y])
c112 = MAE(m49[y],d44[y])


# d - same sample & mounting
c113 = MAE(d14[y],d24[y])
c114 = MAE(d24[y],d34[y])
c115 = MAE(d14[y],d34[y])

# d - same sample & mounting
c116 = MAE(d14[y],d44[y])
c117 = MAE(d24[y],d54[y])
c118 = MAE(d34[y],d64[y])

# d different sample
c119 = MAE(d14[y],d15[y])
c120 = MAE(d16[y],d44[y])
c121 = MAE(d45[y],d46[y])

# n og d - same sample
c122 = MAE(d14[y],n14[y])
c123 = MAE(d15[y],n15[y])
c124 = MAE(d16[y],n16[y])

# d and e - same number
c125 = MAE(d14[y],e14[y])
c126 = MAE(d44[y],e44[y])
c127 = MAE(d15[y],e15[y])
c128 = MAE(d45[y],e45[y])
c129 = MAE(d16[y],e16[y])
c130 = MAE(d46[y],e46[y])





print("e same sample and mounting:")
print(mean([c53,c54,c55]))
print("c same sample and mounting:")
print(mean([c50,c51,c52]))
print("m same sample and mounting:")
print(mean([c92,c93,c94]))

print("   ")

print("c og d different samples:")
print(mean([c01,c02,c2,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c015,c016]))
print("c og e different samples:") 
print(mean([c31,c32,c33,c34,c35,c36,c37,c38,c39,c40,c41,c42,c43,c44,c45,c46,c47,c48]))
print("c og m different samples:")
print(mean([c666,c777,c888,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,c26,c27,c28,c29,c30]))
print("e og m different samples:")
print(mean([c68,c69,c70,c71,c72,c73]))
print("e og n - same samp number")
print(mean([c74,c75,c76,c77,c78,c79]))
print("c og m - same samp number")
print(mean([c80,c81,c82,c83,c84,c85]))
print("  ")
print("e og d - different number:")
print(mean([c86,c87,c88,c89,c90,c91]))

print("c same sample different mounting:")
print(mean([c56,c57,c58,c59,c60,c61]))
print("e same sample different mounting:")
print(mean([c62,c63,c64]))
print("m same sample different mounting:")
print(mean([c65,c66,c67]))


print("c different samples")
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

print("c og d -same sample")
print(mean([c122,c123,c124]))

print("d og e - same number")
print(mean([c125,c126,c127,c128,c129,c130]))












