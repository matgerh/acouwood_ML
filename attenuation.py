import os
import pandas as pd
from math import sqrt
import mlflow
import matplotlib.pyplot as plt

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
c01 = n11[y].subtract(d14[y]).mean()
c02 = n41[y].subtract(d44[y]).mean()
c1 = n12[y].subtract(d15[y]).mean()
c2 = n42[y].subtract(d45[y]).mean()
c4 = n13[y].subtract(d16[y]).mean()
c5 = n43[y].subtract(d46[y]).mean()

c6 = n14[y].subtract(d14[y]).mean()
c7 = n44[y].subtract(d44[y]).mean()
c8 = n15[y].subtract(d15[y]).mean()
c9 = n45[y].subtract(d45[y]).mean()
c10 = n16[y].subtract(d16[y]).mean()
c11 = n46[y].subtract(d46[y]).mean()

c12 = n17[y].subtract(d14[y]).mean()
c13 = n47[y].subtract(d44[y]).mean()
c14 = n18[y].subtract(d15[y]).mean()
c15 = n48[y].subtract(d45[y]).mean()
c015 = n19[y].subtract(d16[y]).mean()
c016 = n49[y].subtract(d46[y]).mean()

# n and e - different samples
c31 = n11[y].subtract(e14[y]).mean()
c32 = n41[y].subtract(e44[y]).mean()
c33 = n12[y].subtract(e15[y]).mean()
c34 = n42[y].subtract(e45[y]).mean()
c35 = n13[y].subtract(e16[y]).mean()
c36 = n43[y].subtract(e46[y]).mean()

c37 = n14[y].subtract(e14[y]).mean()
c38 = n44[y].subtract(e44[y]).mean()
c39 = n15[y].subtract(e15[y]).mean()
c40 = n45[y].subtract(e45[y]).mean()
c41 = n16[y].subtract(e16[y]).mean()
c42 = n46[y].subtract(e46[y]).mean()

c43 = n17[y].subtract(e14[y]).mean()
c44 = n47[y].subtract(e44[y]).mean()
c45 = n18[y].subtract(e15[y]).mean()
c46 = n48[y].subtract(e45[y]).mean()
c47 = n19[y].subtract(e16[y]).mean()
c48 = n49[y].subtract(e46[y]).mean()


# n and e - different samples
c31 = n11[y].subtract(e14[y]).mean()
c32 = n41[y].subtract(e44[y]).mean()
c33 = n12[y].subtract(e15[y]).mean()
c34 = n42[y].subtract(e45[y]).mean()
c35 = n13[y].subtract(e16[y]).mean()
c36 = n43[y].subtract(e46[y]).mean()

c37 = n14[y].subtract(e15[y]).mean()
c38 = n44[y].subtract(e45[y]).mean()
c39 = n15[y].subtract(e16[y]).mean()
c40 = n45[y].subtract(e46[y]).mean()
c41 = n16[y].subtract(e15[y]).mean()
c42 = n46[y].subtract(e45[y]).mean()

c43 = n17[y].subtract(e14[y]).mean()
c44 = n47[y].subtract(e44[y]).mean()
c45 = n18[y].subtract(e15[y]).mean()
c46 = n48[y].subtract(e45[y]).mean()
c47 = n19[y].subtract(e16[y]).mean()
c48 = n49[y].subtract(e46[y]).mean()


# n and m different samples
c16 = n11[y].subtract(m17[y]).mean()
c17 = n41[y].subtract(m47[y]).mean()
c18 = n12[y].subtract(m18[y]).mean()
c16 = n42[y].subtract(m48[y]).mean()
c17 = n13[y].subtract(m19[y]).mean()
c18 = n43[y].subtract(m49[y]).mean()

c19 = n14[y].subtract(m17[y]).mean()
c20 = n44[y].subtract(m47[y]).mean()
c21 = n15[y].subtract(m18[y]).mean()
c22 = n45[y].subtract(m48[y]).mean()
c23 = n16[y].subtract(m19[y]).mean()
c24 = n46[y].subtract(m49[y]).mean()

c25 = n17[y].subtract(m18[y]).mean()
c26 = n47[y].subtract(m48[y]).mean()
c27 = n18[y].subtract(m19[y]).mean()
c28 = n48[y].subtract(m49[y]).mean()
c29 = n19[y].subtract(m17[y]).mean()
c30 = n49[y].subtract(m47[y]).mean()

# e and m
c68 = e14[y].subtract(m17[y]).mean()
c69 = e44[y].subtract(m47[y]).mean()
c70 = e15[y].subtract(m18[y]).mean()
c71 = e45[y].subtract(m48[y]).mean()
c72 = e16[y].subtract(m19[y]).mean()
c73 = e46[y].subtract(m49[y]).mean()

# d og e - different samples
c86 = e14[y].subtract(d15[y]).mean()
c87 = e44[y].subtract(d45[y]).mean()
c88 = e15[y].subtract(d16[y]).mean()
c89 = e45[y].subtract(d46[y]).mean()
c90 = e16[y].subtract(d14[y]).mean()
c91 = e46[y].subtract(d44[y]).mean()

# n - same sample & mounting 
c50 = n11[y].subtract(n21[y]).mean()
c51 = n21[y].subtract(n31[y]).mean()
c52 = n11[y].subtract(n31[y]).mean()

# e - same sample & mounting
c53 = e14[y].subtract(e24[y]).mean()
c54 = e24[y].subtract(e34[y]).mean()
c55 = e14[y].subtract(e34[y]).mean()

# m - same sample & mounting
c92 = m17[y].subtract(m27[y]).mean()
c93 = m27[y].subtract(m37[y]).mean()
c94 = m37[y].subtract(m17[y]).mean()

# n same sample - different mounting
c56 = n11[y].subtract(n41[y]).mean()
c57 = n21[y].subtract(n51[y]).mean()
c58 = n31[y].subtract(n61[y]).mean()

c59 = n17[y].subtract(n47[y]).mean()
c60 = n27[y].subtract(n57[y]).mean()
c61 = n37[y].subtract(n67[y]).mean()

# e same sample - different mounting
c62 = e14[y].subtract(e44[y]).mean()
c63 = e24[y].subtract(e54[y]).mean()
c64 = e34[y].subtract(e64[y]).mean()

# m same sample - different mounting
c65 = m17[y].subtract(m47[y]).mean()
c66 = m27[y].subtract(m57[y]).mean()
c67 = m37[y].subtract(m67[y]).mean()

# n and e - same number
c74 = n14[y].subtract(e14[y]).mean()
c75 = n44[y].subtract(e44[y]).mean()
c76 = n15[y].subtract(e15[y]).mean()
c77 = n45[y].subtract(e45[y]).mean()
c78 = n16[y].subtract(e16[y]).mean()
c79 = n46[y].subtract(e46[y]).mean()

# n and m - same number
c80 = n17[y].subtract(m17[y]).mean()
c81 = n47[y].subtract(m47[y]).mean()
c82 = n18[y].subtract(m18[y]).mean()
c83 = n48[y].subtract(m48[y]).mean()
c84 = n19[y].subtract(m19[y]).mean()
c85 = n49[y].subtract(m19[y]).mean()

# n - different sample number
c95 = n11[y].subtract(n12[y]).mean()
c96 = n13[y].subtract(n14[y]).mean()
c97 = n14[y].subtract(n15[y]).mean()

c98 = n17[y].subtract(n18[y]).mean()
c99 = n19[y].subtract(n41[y]).mean()
c100 = n43[y].subtract(n67[y]).mean()

# m different sample
c101 = m17[y].subtract(m48[y]).mean()
c102 = m47[y].subtract(m49[y]).mean()
c103 = m19[y].subtract(m18[y]).mean()

# e different sample
c104 = e14[y].subtract(e15[y]).mean()
c105 = e16[y].subtract(e44[y]).mean()
c106 = e45[y].subtract(e46[y]).mean()


# d og m - different sample
c107 = m17[y].subtract(d15[y]).mean()
c108 = m47[y].subtract(d45[y]).mean()
c109 = m18[y].subtract(d16[y]).mean()
c110 = m48[y].subtract(d46[y]).mean()
c111 = m19[y].subtract(d14[y]).mean()
c112 = m49[y].subtract(d44[y]).mean()


# d - same sample & mounting
c113 = d14[y].subtract(d24[y]).mean()
c114 = d24[y].subtract(d34[y]).mean()
c115 = d14[y].subtract(d34[y]).mean()

# d - same sample & mounting
c116 = d14[y].subtract(d44[y]).mean()
c117 = d24[y].subtract(d54[y]).mean()
c118 = d34[y].subtract(d64[y]).mean()

# d different sample
c119 = d14[y].subtract(d15[y]).mean()
c120 = d16[y].subtract(d44[y]).mean()
c121 = d45[y].subtract(d46[y]).mean()

# n og d - same sample
c122 = d14[y].subtract(n14[y]).mean()
c123 = d15[y].subtract(n15[y]).mean()
c124 = d16[y].subtract(n16[y]).mean()

# d and e - same number
c125 = d14[y].subtract(e14[y]).mean()
c126 = d44[y].subtract(e44[y]).mean()
c127 = d15[y].subtract(e15[y]).mean()
c128 = d45[y].subtract(e45[y]).mean()
c129 = d16[y].subtract(e16[y]).mean()
c130 = d46[y].subtract(e46[y]).mean()





print("e same sample and mounting:")
print(c53,c54,c55)
print("n same sample and mounting:")
print(c50,c51,c52)
print("m same sample and mounting:")
print(c92,c93,c94)

print("   ")

print("n og d:")
print(c01,c02,c2,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c015,c016)
print("n og e:") 
print(c31,c32,c33,c34,c35,c36,c37,c38,c39,c40,c41,c42,c43,c44,c45,c46,c47,c48)
print("n og m:")
print(c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,c26,c27,c28,c29,c30)
print("e og m:")
print(c68,c69,c70,c71,c72,c73)
print("e og n - same number")
print(c74,c75,c76,c77,c78,c79)
print("n og m - same number")
print(c80,c81,c82,c83,c84,c85)
print("  ")
print("e og d - different number:")
print(c86,c87,c88,c89,c90,c91)

print("n same sample different mounting:")
print(c56,c57,c58,c59,c60,c61)
print("e same sample different mounting:")
print(c62,c63,c64)
print("m same sample different mounting:")
print(c65,c66,c67)


print("n different samples")
print(c95,c96,c97,c98,c99,c100)

print("m different samples")
print(c101,c102,c103)

print("e different samples")
print(c104,c105,c106)

print("d og m - differnt samples")
print(c107,c108,c109,c110,c111,c112)

print("d same sample and mounting")
print(c113,c114,c115)

print("d same sample different mounting")
print(c116,c117,c118)

print("d different samples")
print(c119,c120,c121)

print("n og d -same sample")
print(c122,c123,c124)

print("d og e - same number")
print(c125,c126,c127,c128,c129,c130)
exit()












