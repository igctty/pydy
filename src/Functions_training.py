# coding: UTF-8

import numpy as np
import chainer
from chainer import Variable,Chain
import chainer.links as L
import chainer.functions as F

#sample_array = np.array([[1, 2, 3]], dtype=np.float32)
sample_array = np.array([[1, 2, 3], [-1, 0, 2]], dtype=np.float32)
x = Variable(sample_array)

y = F.sum(x)
print(y.data)

y2 = F.average(x)
print(y2.data)

y3 = F.max(x)
print(y3.data)

y4 = F.sigmoid(x)
print(y4.data)

y5 = F.relu(x)
print(y5.data)