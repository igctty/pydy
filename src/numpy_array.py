# coding: UTF-8

import numpy as np

a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(a)

b = np.arange(10)
print(b)

c = np.array([[0, 1, 2], [3, 4, 5]])
print(c)

d = np.arange(6).reshape(2, 3)
print(d)

e = np.arange(6).reshape(3, 2)
print(e)

f = np.arange(12).reshape(2, 3, 2)
print(f)

g = np.arange(42).reshape(6, 7)
print(np.shape(g))
print(np.size(g))

(row, col) = np.shape(g)
print(row)
print(col)
h = np.zeros(10)
print(h)

i = np.ones(10)
print(i)

j = np.random.rand(10)
print(j)

k = np.random.permutation(range(10))
print(k)

l = np.arange(6).reshape(2, 3)
m = np.arange(6).reshape(2, 3)
print(l)
print(m)

n = np.hstack([l, m])
print(n)

o = np.vstack([l, m])
print(o)

p = np.arange(20).reshape(4, 5)
print(p)
q = p[[0,2], :]
print(q)

r = p[:, [0, 2, 4]]
print(r)

s = p
s[s%2 == 0] = 0
print(s)

####################
print("---------------------------------")
aa = np.arange(6).reshape(2, 3)
print(aa)

bb = aa + 1
print(bb)

cc = aa * 2
print(cc)

dd = aa ** 2
print(dd)

ee = np.sum(aa)
print(ee)

ff = np.mean(aa)
print(ff)

gg = np.arange(1, 7).reshape(2, 3)
print(gg)

hh = aa + gg
print(hh)

ii = aa * gg
print(ii)