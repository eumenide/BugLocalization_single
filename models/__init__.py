#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np

if __name__ == '__main__':
	a = np.zeros(5)

	a[0] = 1
	a[1] = 2
	a[2] = 3
	a[3] = 4
	a[4] = 5

	a /= 2

	print(a)

# a = [[3, 1, 2, 4], [6, 5, 8, 7], [9, 1, 5, 3]]
# b = [[0.3, 0.1, 0.2, 0.4], [0.6, 0.5, 0.8, 0.7], [0.9, 0.1, 0.5, 0.3]]
#
# a = np.array(a)
# b = np.array(b)
#
# # print(a)
# # print(b)
# for i in range(0, len(a)):
# 	# print(a[i])
# 	sorted_indices = np.argsort(-a[i], axis=0)
# 	a[i] = a[i][sorted_indices]
# 	b[i] = b[i][sorted_indices]
# 	print(a[i])
# 	for j in range(0, len(b[i])):
# 		print(b[i][j])

# print(b[i])
# print(a)
# print(b)

# print(a)
# sorted_indices = np.argsort(-a, axis=1)
# print(sorted_indices)
# sorted_a = a[:,sorted_indices]
# print(sorted_a)
