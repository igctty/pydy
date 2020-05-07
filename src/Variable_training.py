# coding: UTF-8

import chainer
from chainer import Variable
import numpy as np

# numpyの配列を作成
input_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
print(input_array)

# Variableオブジェクトを作成
x = Variable(input_array)
print(x.data)

# 計算
# y = x * 2 + 1
y = x ** 2 + 2 * x +1 # y'=2*ｘ+2
print(y.data)

# 微分値を求める
# 要素が複数の場合は xxx.grad に初期値が必要
y.grad = np.ones((2, 3), dtype=np.float32)
# y→x と遡って微分値を求める
y.backward()
# y=ｘ*2+1 の微分形は y=2
print(x.grad)