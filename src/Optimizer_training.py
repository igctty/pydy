# coding: UTF-8

import numpy as np
import chainer
from chainer import Variable,Chain
import chainer.links as L
import chainer.functions as F
import chainer.optimizers as O

# model
class MyChain(Chain):
    def __init__(self):
        super().__init__(
            l1 = L.Linear(1,2),
            l2 = L.Linear(2,1),
        )

    def __call__(self, x):
        h = F.sigmoid(self.l1(x))
        return self.l2(h)

# Optimizer
model = MyChain()
optimizer = O.SGD() # 最適化アルゴリズム：SGD=確率的降下法
# optimizer = O.Adam() # 最適化アルゴリズム：Adam
optimizer.setup(model)

# execution
input_array = np.array([[1]], dtype=np.float32)
answer_array = np.array([[1]], dtype=np.float32)
x = Variable(input_array)
t = Variable(answer_array)

model.cleargrads() #model 勾配初期化
y=model(x)

loss=F.mean_squared_error(y,t) #二乗誤差 y t の誤差を求める。
loss.backward() #誤差の逆伝播

# 前後比較
print(model.l1.W.data)
optimizer.update()
print(model.l1.W.data)