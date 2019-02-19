#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import logging
import json
import numpy as np
import pandas as pd


logger_main = None
main_dir = '../datasets/'
project_name = 'aspectj'
project_prefix = 'org.aspectj/'

ratio_train = 0.6
ratio_test = 0.2
ratio_valid = 0.2

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


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
		elif isinstance(obj. np.ndarray):
			return obj.tolist()
		else:
			return super(MyEncoder, self).default(obj)


def load_label(project):
	# 读入label数据
	label_file = main_dir + project + '/' + project + '_label.json'
	project_label = pd.read_json(label_file, orient='records', dtype=False)

	return project_label


def load_report_vec(project):
	# 读入report_vec数据
	report_file = main_dir + project + '/' + project + '_report_vec.json'

	report_vec = pd.read_json(report_file, orient='records', dtype=False)

	return report_vec


def concatenate_data(label, report):
	# 合并数据
	new_data = pd.merge(label, report, on=['bug_id'])

	return new_data


def split_random(project, data):
	# todo 随机划分数据集
	train_file = main_dir + project + '/' + project + '_train.json'
	test_file = main_dir + project + '/' + project + '_test.json'
	valid_file = main_dir + project + '/' + project + '_valid.json'

	cnt_valid = round(len(data) * ratio_valid, 0)
	cnt_test = round(len(data) * ratio_test, 0)
	cnt_train = len(data) - cnt_valid - cnt_test

	# 随机打乱数据集
	np.random.shuffle(data)

	return


def split_time(project, data):
	# todo 按时间划分数据集

	return


def run_main(project):
	#

	# 1 读取_label.json数据
	logger_main.info('load label data from file......')
	project_label = load_label(project)

	# 2 读取_report_vec.json数据
	logger_main.info('load report vectors from file.......')
	report_vec = load_report_vec(project)

	# 3 根据bug_id拼接数据
	logger_main.info('concatenate data according to bug_id......')
	origin_data = concatenate_data(project_label, report_vec)

	# 4 随机划分数据集
	logger_main.info('split datasets randomly......')
	split_random(project, origin_data)

	# 5 按时间划分数据集
	logger_main.info('split datasets according to time......')
	split_time(project, origin_data)

	return


if __name__ == '__main__':
	arr = np.arange(18).reshape((9, 2))
	print(arr)
	np.random.shuffle(arr)
	print(arr)
	# tmp_a = load_label(project_name)
	# print(len(tmp_a))
	#
	# tmp_b = load_report_vec(project_name)
	# print(len(tmp_b))
	#
	# tmp_c = concatenate_data(tmp_a, tmp_b)
	# print(len(tmp_c))
	#
	# print(tmp_c.loc[0])
	# np.random.shuffle(tmp_c)
	# print(tmp_c.loc[0])
	#
	# cnt_valid = round(len(tmp_c) * ratio_valid, 0)
	# print(cnt_valid)
	# cnt_test = round(len(tmp_c) * ratio_test, 0)
	# print(cnt_test)
	# cnt_train = len(tmp_c) - cnt_valid - cnt_test
	# print(cnt_train)
	# logger_main = get_logger('run_main', main_dir + project_name + '/' + 'dataset_main.log')
	# logger_main.info('split datasets starting.............')
	#
	# run_main(project_name)
	#
	# logger_main.info('split datasets ended successfully ^_^')
