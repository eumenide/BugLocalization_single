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
project_prefix = 'org.aspectj/'

file_list = []
bug_list = []


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


temp = 1


def split_file(data, commit, bug_id, logger):
	'''

	:param logger:
	:param bug_id:
	:param commit:
	:param data:
	:return:
	'''

	global temp
	# 3.1 以.java分隔得到file的列表
	result = []
	data = data.split('.java')

	logger.info(str(temp) + ' / 593')
	for x in data:
		if x in [' ', '']:
			continue

		# 3.2 组合得到新文件名
		file = project_prefix + (os.path.dirname(x) + '/' + commit + ' ' + os.path.basename(x)).strip()

		# 3.3 过滤掉file_list中没有的文件
		if file.strip() in file_list:
			result.append(file.strip())

	# 3.4 将没有相关文件的bug的bug_id记录
	if len(result) == 0:
		bug_list.append(bug_id)
		logger.info(str(bug_id))

	# 3.5 根据与bug相关的文件id产生于该bug相关文件标签
	logger.info('generate label list......')
	label = map(lambda item: 1 if item in result else 0, file_list)

	temp += 1

	return list(label)


def generate_label(data, project):
	#
	logger = get_logger('generate label', main_dir + project + '/' + 'generate_label.log')

	data['label'] = data.apply(lambda x: split_file(x['files'], x['commit'], x['bug_id'], logger), axis=1)

	data.drop(labels=['files', 'commit'], axis=1, inplace=True)

	# 剔除需要剔除的bug
	data = data[~data['bug_id'].isin(bug_list)]

	return data


def save_label(project, data):
	#
	label_file = main_dir + project + '/' + project + '_label.json'
	data.to_json(label_file, orient='records')

	extra_bug_file = main_dir + project + '/' + project + '_bug.json'
	with open(extra_bug_file, 'w') as f:
		json.dump(bug_list, f)

	return


def run_main(project):
	#
	xlsx_file = main_dir + project + '/' + project + '.xlsx'

	# 1 读取数据
	logger_main.info('load data from xsl......')
	nlp_data = load_data_from_xsl(xlsx_file)

	# 2 读入文件列表
	logger_main.info('load file list from file......')
	load_file_list(project)

	# 3 处理每个bug的files，得到label
	logger_main.info('generate label for bug report and source code......')
	nlp_data = generate_label(nlp_data, project)

	# 4 将结果存储在_label.json中
	logger_main.info('save project label to file......')
	save_label(project, nlp_data)

	print(len(nlp_data))

	return


if __name__ == '__main__':
	# tmp_a = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
	# tmp_b = ['b', 'e', 'f', 'j']
	# tmp_c = map(lambda x: 1 if x in tmp_b else 0, tmp_a)
	# print(list(tmp_c))
	logger_main = get_logger('run_main', main_dir + project_name + '/' + 'label_main.log')
	logger_main.info('generate label starting.............')

	run_main(project_name)

	logger_main.info('generate label ended successfully ^_^')
