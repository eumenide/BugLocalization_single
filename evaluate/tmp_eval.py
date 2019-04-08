#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import logging
import pandas as pd
import json
import numpy as np

from keras.utils import to_categorical


main_dir = '../datasets/'
project = 'aspectj'
num = 1405


class MyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.integer):
			return int(obj)
		elif isinstance(obj, np.floating):
			return float(obj)
		elif isinstance(obj, np.ndarray):
			return obj.tolist()
		else:
			return super(MyEncoder, self).default(obj)


def load_data():
	# 读取数据
	predict_file = main_dir + project + '/' + project + '_predict_v30.json'
	# predict_data = pd.read_json(predict_file, orient='index', dtype=False)
	with open(predict_file, 'r') as json_data:
		data = json.load(json_data)
	# pd.DataFrame.from_dict(data, orient='index').T.set_index('index')
	# print(data['predict:'])
	print('predict')
	print(len(data['predict:']))
	totals = 0
	for i in range(len(data['predict:'])):
		tmp = data['predict:'][i]
		if tmp[1] > tmp[0]:
			print(str(i) + "  :  " + str(tmp))
			totals += 1
	print('totals :  ' + str(totals))

	print('real')
	print(len(data['real:']))
	totals = 0
	for i in range(len(data['real:'])):
		tmp = data['real:'][i]
		if tmp[1] > tmp[0]:
			print(str(i) + "  :  " + str(tmp))
			totals += 1
	print('totals :  ' + str(totals))

	# print(data['real:'])
	# print(len(data['real:']))
	# return predict_data['predict:'], predict_data['real:']


def load_train_data():
	# 读取数据
	predict_file = main_dir + project + '/' + project + '_valid.json'
	data = pd.read_json(predict_file, orient='records', dtype=False)
	# with open(predict_file, 'r') as json_data:
	# 	data = json.load(json_data)
	# pd.DataFrame.from_dict(data, orient='index').T.set_index('index')
	# print(data['predict:'])
	print(type(data))
	print(data.head(10))
	print(type(data['label']))
	print('predict')
	print(len(data['label']))

	labels = np.empty((0, 2))
	for i in range(len(data['label'])):
		label = data['label'][i]
		label = to_categorical(label, num_classes=2)
		label = label.reshape((-1,2))
		labels = np.concatenate((labels, label), axis=0)

	totals = 0
	for i in range(len(labels)):
		tmp = labels[i]
		if tmp[1] > tmp[0]:
			totals += 1
	print('totals :  ' + str(totals))


def cal_map(y_pred, y_real):
	# 计算the mean average of the precision

	predY = np.array(y_pred)
	realY = np.array(y_real)
	mAP = 0

	for i in range(0, len(predY)):
		predY[i] = np.array(predY[i])
		realY[i] = np.array(realY[i])
		sorted_indices = np.argsort(-predY[i], axis=0)
		predY[i] = predY[i][sorted_indices]
		realY[i] = realY[i][sorted_indices]

		t_i = 0
		sum_i = 0
		for j in range(0, len(realY[i])):
			ind_j = realY[i][j]
			t_i += ind_j
			sum_i += t_i / (j + 1) * ind_j

		avg_i = sum_i / t_i
		mAP += avg_i

	mAP /= len(predY)

	return mAP


def cal_mrr(y_pred, y_real):
	# 计算 the mean of the Reciprocal Rank
	predY = np.array(y_pred)
	realY = np.array(y_real)

	mrr = 0
	for i in range(0, len(predY)):
		predY[i] = np.array(predY[i])
		realY[i] = np.array(realY[i])
		sorted_indices = np.argsort(-predY[i], axis=0)
		predY[i] = predY[i][sorted_indices]
		realY[i] = realY[i][sorted_indices]

		for j in range(0, len(realY[i])):
			if realY[i][j] == 1:
				mrr += 1 / (j + 1)
				break

	mrr = mrr / len(predY)

	return mrr


def cal_accuracy_k(y_pred, y_real):
	# 计算Accuracy@k
	predY = np.array(y_pred)
	realY = np.array(y_real)
	acc_k = np.zeros(20)

	for i in range(0,len(predY)):
		predY[i] = np.array(predY[i])
		realY[i] = np.array(realY[i])
		sorted_indices = np.argsort(-predY[i], axis=0)
		predY[i] = predY[i][sorted_indices]
		realY[i] = realY[i][sorted_indices]

		print(predY[i])

		for j in range(0, 20):
			if realY[i][j] == 1:
				for k in range(j, 20):
					acc_k[k] += 1
				break
	acc_k /= num

	return list(acc_k)


def save_evaluation(MAP, MRR, ACC_K):
	# 保存评估值

	evaluation_file = main_dir + project + '/' + project + '_eval_v10_4.json'
	result = {"MAP": MAP, "MRR": MRR, "ACC_K": ACC_K}
	with open(evaluation_file, 'w') as f:
		json.dump(result, f, cls=MyEncoder)

	return


def run_main():
	#
	# y_pred, y_real = load_data()
	# load_data()
	load_train_data()
	# print(y_pred.head)
	return


if __name__ == '__main__':
	run_main()

