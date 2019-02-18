#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import logging
import json
import os
import pandas as pd


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

logger_main = None
main_dir = '../datasets/'
project_name = 'aspectj'

file_list = []


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


def load_data_from_xsl(file_name):
	nlp_data = pd.read_excel(file_name, sheet_name=0, header=0, usecols=[1, 5, 7, 9],
	                         converters={'bug_id': str, 'time': str, 'commit': str, 'files': str})
	nlp_data.fillna(' ', inplace=True)

	return nlp_data


def load_file_list(project):
	# 加载文件列表
	global file_list
	input_file = main_dir + project + '/' + project + '.txt'
	with open(input_file, 'r') as f:
		file_list = f.readlines()

	file_list = list(map(lambda x: x.strip(), file_list))

	return


def split_file(data, commit):
	'''

	:param commit:
	:param data:
	:return:
	'''

	# 3.1 以.java分隔得到file的列表
	result = []
	data = data.split('.java')
	for x in data:
		if x in [' ', '']:
			continue

		# 3.2 组合得到新文件名
		file = os.path.dirname(x) + '/' + commit + ' ' + os.path.basename(x)
		if file.strip() not in file_list:
			file_list.append(file.strip())
		result.append(file.strip())

	return result


def generate_label(data, project):
	# todo
	logger = get_logger('generate label', main_dir + project_name + '/' + 'generate_label.log')

	data['files_new'] = data.apply(lambda x: split_file(x['files'], x['commit']), axis=1)

	data.drop(labels=['files', 'commit'], axis=1, inplace=True)

	return data


def save_label(project, data):
	#
	data.drop(labels=['commit', 'files'], axis=1, inplace=True)

	label_file = main_dir + project + '/' + project + '_label.json'
	data.to_json(label_file, orient='records')

	return


def run_main(project):
	#
	xlsx_file = main_dir + project_name + '/' + project_name + '.xlsx'

	# 1 读取数据
	logger_main.info('load data from xsl......')
	nlp_data = load_data_from_xsl(xlsx_file)

	# 2 读入文件列表
	logger_main.info('load file list from file......')
	load_file_list(project_name)

	# 3 处理每个bug的files，得到label
	logger_main.info('generate label for bug report and source code......')
	nlp_data = generate_label(nlp_data, project_name)

	# 4 将结果存储在_label.json中
	logger_main.info('save project label to file......')
	save_label(project_name, nlp_data)

	return


if __name__ == '__main__':
	tmp = 'org.aspectj/ajbrowser/src/org/aspectj/tools/ajbrowser/85a827a BrowserProperties '
	load_file_list(project_name)
	if tmp in file_list:
		print(True)
	else:
		print(False)
	# logger_main = get_logger('run_main', main_dir + project_name + '/' + 'label_main.log')
	# logger_main.info('generate label starting.............')
	#
	# run_main(project_name)
	#
	# logger_main.info('generate label ended successfully ^_^')
