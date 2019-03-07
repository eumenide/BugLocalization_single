#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import numpy as np
from keras.utils import to_categorical

a = [[0, 1, 1], [1, 0, 0], [0, 0, 1]]

b = [[1, 0], [0, 1]]
a = np.array(a)
print(to_categorical(a, 2))
b = np.array(b)
c = np.array(a)[np.newaxis, :]
for i in range(3):
	d = a[np.newaxis, :]
	c = np.vstack((c, d))
# print(a)
# print(c)

