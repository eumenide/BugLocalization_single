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
		elif isinstance(obj.np.ndarray):
			return obj.tolist()
		else:
			return super(MyEncoder, self).default(obj)


def load_label(project):
	# 读入label数据
	label_file = main_dir + project + '/' + project + '_label.json'
	project_label = pd.read_json(label_file, orient='records', dtype=False)
	project_label.rename(columns={'report_timestamp': 'time'}, inplace=True)

	return project_label


def load_report_vec(project):
	# 读入report_vec数据
	report_file = main_dir + project + '/' + project + '_report_vec.json'

	report_vec = pd.read_json(report_file, orient='records', dtype=False)

	return report_vec


def concatenate_data(label, report):
	# 合并数据
	logger = get_logger('concatenate data', main_dir + project_name + '/conca_data.log')
	logger.info('concatenate data starting......')
	logger.info('origin label data #    ' + str(len(label)))
	logger.info('origin report data #    ', str(len(report)))

	new_data = pd.merge(label, report, on=['bug_id'])
	logger.info('data # after merge    ' + str(len(new_data)))

	new_data.reset_index(drop=True, inplace=True)
	logger.info('data # after reset index    ' + str(len(new_data)))

	return new_data


def split_random(project, data):
	# 随机划分数据集
	logger = get_logger('split datasets randomly', main_dir + project + '/' + 'split_random.log')
	logger.info('split datasets randomly starting......')

	train_file = main_dir + project + '/' + project + '_train.json'
	test_file = main_dir + project + '/' + project + '_test.json'
	valid_file = main_dir + project + '/' + project + '_valid.json'

	cnt_valid = round(len(data) * ratio_valid, 0)
	cnt_test = round(len(data) * ratio_test, 0)
	cnt_train = len(data) - cnt_valid - cnt_test

	logger.info('train data #     ' + str(cnt_train))
	logger.info('test data #      ' + str(cnt_test))
	logger.info('valid data #      ' + str(cnt_valid))

	# 随机打乱数据集
	data = data.sample(frac=1.0)
	data = data.reset_index(drop=True)
	train_data = data.loc[0: (cnt_train - 1)]
	test_data = data.loc[cnt_train: (cnt_train + cnt_test - 1)]
	valid_data = data.loc[(cnt_test + cnt_train):]

	logger.info('train data #     ' + str(len(train_data)))
	logger.info('test data #      ' + str(len(test_data)))
	logger.info('valid data #      ' + str(len(valid_data)))

	train_data.to_json(train_file, orient='records')
	test_data.to_json(test_file, orient='records')
	valid_data.to_json(valid_file, orient='records')

	return


def split_time(project, data):
	# 按时间划分数据集
	logger = get_logger('split datasets according to time', main_dir + project + '/' + 'split_time.log')
	logger.info('split datasets according to time starting......')

	train_file = main_dir + project + '/' + project + '_train_t.json'
	test_file = main_dir + project + '/' + project + '_test_t.json'
	valid_file = main_dir + project + '/' + project + '_valid_t.json'

	cnt_valid = round(len(data) * ratio_valid, 0)
	cnt_test = round(len(data) * ratio_test, 0)
	cnt_train = len(data) - cnt_valid - cnt_test

	logger.info('train data #     ' + str(cnt_train))
	logger.info('test data #      ' + str(cnt_test))
	logger.info('valid data #      ' + str(cnt_valid))

	# 按时间排序
	data = data.sort_values(by="time", ascending=False)
	data = data.reset_index(drop=True)

	test_data = data.loc[0: cnt_test - 1]
	valid_data = data.loc[cnt_test: cnt_test + cnt_valid - 1]
	train_data = data.loc[cnt_valid + cnt_test:]

	logger.info('train data #     ' + str(len(train_data)))
	logger.info('test data #      ' + str(len(test_data)))
	logger.info('valid data #      ' + str(len(valid_data)))

	train_data.to_json(train_file, orient='records')
	test_data.to_json(test_file, orient='records')
	valid_data.to_json(valid_file, orient='records')

	return


def run_main(project):
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

	logger_main = get_logger('run_main', main_dir + project_name + '/' + 'dataset_main.log')
	logger_main.info('split datasets starting.............')

	run_main(project_name)

	logger_main.info('split datasets ended successfully ^_^')
