# coding: UTF-8

import chainer
from chainer import Variable, Chain, optimizers, serializers, datasets
import chainer.links as L
import chainer.functions as F

from chainer.datasets import tuple_dataset
from chainer import training, iterators
from chainer.training import extensions

import numpy as np

# -- MNISTデータの読み込み --
mnist_data = datasets.get_mnist(ndim=3) #get_mnist：手書き数字のデータセットを取得, ndim=3：1*28*28の三次元画像
train_data = mnist_data[0]
test_data = mnist_data[1]

print("Train:", len(train_data))
print("Test:", len(test_data))

# -- MNIST画像の表示 --
import matplotlib.pyplot as plt

index = 59997
plt.imshow(train_data[index][0].reshape(28, 28), cmap='gray')
plt.title(train_data[index][1])
plt.show()