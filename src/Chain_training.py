# coding: UTF-8

import numpy as np
import chainer
from chainer import Variable,Chain
import chainer.links as L

l1 = L.Linear(4, 3)
l2 = L.Linear(3, 2)

def my_forward(x):
    h = l1(x)
    return l2(h)

input_array = np.array([[1, 2, 3, 4]], dtype=np.float32)
x = Variable(input_array)
y = my_forward(x)
print(y.data)

class MyClass():
    def __init__(self):
        self.l1 = L.Linear(4, 3)
        self.l2 = L.Linear(3, 2)

    def forwad(self, x):
        h = self.l1(x)
        return self.l2(h)


print("--------")

input_array = np.array([[1, 2, 3, 4]], dtype=np.float32)
m = Variable(input_array)
my_class = MyClass()
n = my_class.forwad(m)
print(n.data)

# Chain クラスを継承
class MyChain(Chain):
    def __init__(self):
        super().__init__(
            l1 = L.Linear(4, 3),
            l2 = L.Linear(3, 2),
        )

    def __call__(self, x):
        h = self.l1(x)
        return self.l2(h)

print("--------")

input_array = np.array([[1, 2, 3, 4]], dtype=np.float32)
x1 = Variable(input_array)
my_chain = MyChain()
y1 = my_chain(x1)
print(y1.data)