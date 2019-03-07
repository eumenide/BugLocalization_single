#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import logging
import pandas as pd
import json
import numpy as np

main_dir = '../datasets/'
project = 'aspectj'
num = 1405

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger_main = None


def get_logger(logger_name, log_file):
	'''

	:param logger_name:
	:param log_file:
	:return:
	'''
	logger = logging.getLogger(logger_name)
	logger.setLevel(logging.INFO)

	fh = logging.FileHandler(log_file)
	fmt = '%(asctime)s : %(threadName)s : %(levelname)s : %(message)s'
	formatter = logging.Formatter(fmt)
	fh.setFormatter(formatter)

	logger.addHandler(fh)
	return logger


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
	logger_main.info('load predicted data from file......')
	predict_file = main_dir + project + '/' + project + '_predict_v11.json'
	predict_data = pd.read_json(predict_file, orient='records', dtype=False)

	return predict_data['predict:'], predict_data['real:']


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
		logger_main.info('average precision of #' + str(i + 1) + ' / ' + str(num) + '  :  ' + str(avg_i))

	mAP /= len(predY)
	logger_main.info('MAP of the prediction :    ' + str(mAP))

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
				logger_main.info('the first position of #' + str(i + 1) + ' / ' + str(num) + '  :  ' + str(j + 1))
				break

	mrr = mrr / len(predY)
	logger_main.info('MRR of the prediction :    ' + str(mrr))

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

		logger_main.info('acc_k after #' + str(i + 1) + ' / ' + str(num) + '  :    ' + str(acc_k))

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
	# todo
	logger_main.info('load data from file......')
	y_pred, y_real = load_data()

	# 计算MAP
	logger_main.info('calculate MAP of prediction......')
	MAP = cal_map(y_pred, y_real)

	# 计算MRR
	logger_main.info('calculate MRR of prediction......')
	MRR = cal_mrr(y_pred, y_real)

	# 计算Accuracy@k
	logger_main.info('calculate Accuracy@k of prediction......')
	acc_k = cal_accuracy_k(y_pred, y_real)

	logger_main.info('save evaluation to file......')
	save_evaluation(MAP, MRR, acc_k)

	return


if __name__ == '__main__':
	logger_main = get_logger('run_main', main_dir + project + '/' + 'evaluate_v10_3.log')
	logger_main.info('evaluate model v1.0 starting.............')

	run_main()

	logger_main.info('evaluate model v1.0 ended successfully ^_^')
