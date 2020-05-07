# coding: UTF-8

import numpy as np
from sklearn import datasets
import chainer
from chainer import Variable,Chain
from chainer.datasets import tuple_dataset
from chainer import training,iterators
from chainer.training import extensions
import chainer.links as L
import chainer.functions as F
import chainer.optimizers as O


# データ読み込み
iris_data = datasets.load_iris()
# print(iris_data)

# データの取り出し
x = iris_data.data.astype(np.float32) # iris 花の特徴を表す4種のデータ
t = iris_data.target # 品種を表す数値
n = t.size # 品種を表す数値のサイズ
# print(x)
# print(t)
# print(n)

# 教師データの準備
t_matrix = np.zeros(3*n).reshape(n, 3).astype(np.float32)
for i in range(n):
    t_matrix[i,t[i]] = 1.0

# print(t_matrix)

# 訓練用データとテスト用データ　半分ずつ
indexes = np.arange(n)
indexes_training = indexes[indexes%2 != 0]
indexes_test = indexes[indexes%2 == 0]

# print(indexes)
# print(indexes_training)
# print(indexes_test)

x_training = x[indexes_training, : ] # 訓練用 入力
t_training = t_matrix[indexes_training, : ] # 訓練用 正解
x_test = x[indexes_test, : ] # テスト用 入力
t_test = t[indexes_test] # テスト用 正解

# print(x_training)
# print(x_test)
# print(t_training)
# print(t_test)

# # Variable に変換
# x_training_v = Variable(x_training)
# t_training_v = Variable(t_training)
x_test_v = Variable(x_test)
# print(x_training_v)
# print(t_training_v)
# print(x_test_v)

# trainer
train = tuple_dataset.TupleDataset(x_training, t_training)

# Chain
class IrisChain(Chain):
    def __init__(self):
        super(IrisChain, self).__init__(
            l1=L.Linear(4, 6),
            l2=L.Linear(6, 6),
            l3=L.Linear(6, 3),
        )

    def __call__(self, x, t):
        return F.mean_squared_error(self.predict(x), t)

    def predict(self,x):
        h1 = F.sigmoid(self.l1(x))
        h2 = F.sigmoid(self.l2(h1))
        h3 = self.l3(h2)
        return h3

# model, optimizer
model = IrisChain()
optimizer = O.Adam()
optimizer.setup(model)

# learn
# for i in range(10000):
#     model.cleargrads()
#     y_training_v = model.predict(x_training_v)
#
#     # 損失関数：平均二乗誤差
#     loss = F.mean_squared_error(y_training_v, t_training_v)
#     loss.backward()
#
#     # 重みの更新
#     optimizer.update()
train_iter = iterators.SerialIterator(train, 30) #ミニバッチデータの数：30個
updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (5000, 'epoch'))
trainer.extend(extensions.ProgressBar())
trainer.run()


# テスト
model.cleargrads()
y_test_v = model.predict(x_test_v)
y_test = y_test_v.data

# 正解数カウント
correct = 0
rowCount = y_test.shape[0] # y_test の要素数
for i in range(rowCount):
    maxIndex = np.argmax(y_test[i, :]) # np.argmax関数は最大の要素のインデックスを返す
    print(y_test[i, :], maxIndex)
    if maxIndex == t_test[i]:
        correct = correct+1

# 正解率
print("Correct:", correct, "Total:", rowCount, "Accuracy:", (correct/rowCount)*100, "%")

