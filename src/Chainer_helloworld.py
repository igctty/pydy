# coding: UTF-8

import numpy as np
import matplotlib.pyplot as plt
import chainer
from chainer import Variable,Chain
import chainer.links as L
import chainer.functions as F
import chainer.optimizers as O

# 階段回数のデータ
x,t = [],[]
for i in np.linspace(-1,1,100):
    x.append([i])
    if i<0:
        t.append([0])
    else:
        t.append([1])

# # グラフにプロット
# plt.plot(np.array(x, dtype=np.float32).flatten(), np.array(t, dtype=np.float32).flatten()) #flatten：一重配列
# plt.show()

class MyChain(Chain):
    def __init__(self):
        super().__init__(
            l1 = L.Linear(1,10),
            l2 = L.Linear(10,1),
        )

    def predict(self, x):
        h = F.sigmoid(self.l1(x))
        return self.l2(h)

# x, t を Variable に
x = Variable(np.array(x, dtype=np.float32))
t = Variable(np.array(t, dtype=np.float32))

model = MyChain()
optimizer = O.Adam()
optimizer.setup(model)

# 学習させる 10万回
y = None
for i in range(100000):
    model.cleargrads()
    y = model.predict(x)

    # 途中経過をグラフにプロット 1万回ごと
    if i%10000 == 0:
        plt.plot(x.data.flatten(), y.data.flatten())
        plt.title("i=" + str(i))
        plt.show()

    # 損失関数による誤差の計算　平均二乗誤差
    loss = F.mean_squared_error(y,t)
    loss.backward()

    # 重みの更新
    optimizer.update()

# 終了
plt.plot(x.data.flatten(), y.data.flatten())
plt.title("Finish")
plt.show()