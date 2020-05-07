# coding: UTF-8

import numpy as np
import chainer
from chainer import Variable
import chainer.links as L

# Links の Linear Link （入力に係数をかけたものと、バイアスをすべて足し合わせる）関数によりオブジェクト l を作成
# l = L.Linear(3, 2) # 入力3、出力2
l = L.Linear(6, 4)
print(l.W.data) # l.W が係数 … 重み（結合荷重）
print(l.b.data) # l.b がバイアス

# input_array = np.array([[1, 2, 3]], dtype=np.float32)
# input_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
input_array = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], dtype=np.float32)
x = Variable(input_array)
y = l(x)
print(y.data)

# l の勾配をゼロに初期化
print("--------------")
l.cleargrads()

# y→l と遡って微分の計算
# y.grad = np.ones((1, 2), dtype=np.float32)
# y.grad = np.ones((2, 2), dtype=np.float32)
y.grad = np.ones((4, 4), dtype=np.float32)
y.backward()
print(l.W.grad) #係数の微分値
print(l.b.grad) #バイアスの微分値
