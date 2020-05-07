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
import chainer.serializers as S


# データ読み込み
iris_data = datasets.load_iris()

# データの取り出し
x = iris_data.data.astype(np.float32) # iris 花の特徴を表す4種のデータ
t = iris_data.target # 品種を表す数値
n = t.size # 品種を表す数値のサイズ

# 教師データの準備
t_matrix = np.zeros(3*n).reshape(n, 3).astype(np.float32)
for i in range(n):
    t_matrix[i,t[i]] = 1.0

# 訓練用データとテスト用データ　半分ずつ
indexes = np.arange(n)
indexes_training = indexes[indexes%2 != 0]
indexes_test = indexes[indexes%2 == 0]

x_training = x[indexes_training, : ] # 訓練用 入力
t_training = t_matrix[indexes_training, : ] # 訓練用 正解
x_test = x[indexes_test, : ] # テスト用 入力
t_test = t[indexes_test] # テスト用 正解

# Variable に変換
x_test_v = Variable(x_test)

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
train_iter = iterators.SerialIterator(train, 30) #ミニバッチデータの数：30個
updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (5000, 'epoch'))
trainer.extend(extensions.ProgressBar())
trainer.run()

# model save
S.save_npz("my_iris.npz", model)
