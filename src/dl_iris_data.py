# coding: UTF-8

import numpy as np
from sklearn import datasets
import chainer
from chainer import Variable,Chain
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
