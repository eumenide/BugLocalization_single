#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import numpy as np
from keras.utils import to_categorical

a = [[0, 1], [1, 0], [0, 1]]
b = [[1, 0], [0, 1], [1, 0], [0, 1]]

a = np.array(a)
b = np.array(b)
c = to_categorical(a[0:2], 2)
d = to_categorical(b[0:3], 2)
c = c.reshape((-1,2))
d = d.reshape((-1,2))

print(c)
print(d)
# print(a)
# print(c)

