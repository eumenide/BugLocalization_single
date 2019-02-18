#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import logging
import json
import numpy as np
import nltk
import re

from nltk.corpus import stopwords
from gensim.models import KeyedVectors


total_files = {'aspectj': 1405, 'eclipseUI': 15179, 'jdt': 12682, 'swt': 8119, 'tomcat': 2355}
main_dir = '../datasets/'
project_name = 'aspectj'

stopwords_dic = set(stopwords.words('english')) | {'.', ':', '(', ')', '\n'}


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


def clean_str(string):
	string = re.sub(r"[^a-zA-Z0-9.]", " ", string)
	string = re.sub(r"\'s", " is", string)
	string = re.sub(r"\'ve", " have", string)
	string = re.sub(r"n\'t", "n not", string)
	string = re.sub(r"\'re", " are", string)
	string = re.sub(r"\'d", " would", string)
	string = re.sub(r"\'ll", " will", string)
	string = re.sub(r"\'m", " am", string)
	string = re.sub(r"\s{2,}", " ", string)
	string = re.sub(r"([A-Za-z0-9][a-z])([A-Z])", lambda x: x.group(1) + " " + x.group(2), string)
	string = re.sub(r"([A-Za-z0-9][a-z])([A-Z])", lambda x: x.group(1) + " " + x.group(2), string)
	return string.strip().lower()


def tokenize_and_stopwords(text='', stopworddic=None, pattern=None):
	'''
	对文本进行分词并去除停用词，使用nltk工具
	:param pattern: 分词的正则表达式，有默认值
	:param stopworddic: 提供的stopwords库，默认使用NLTK中的停用词库
	:param text: 待分词的文本，一个字符串
	:return: 分词后的结果，一个数组
	'''
	if stopworddic is None:
		stopworddic = stopwords_dic
	if pattern is None:
		pattern = r"""(?x)
							(?:[A-Z]\.)+
							|\d+(?:\.\d+)+%?
							|\w+(?:[-']\w+)*
							|\.\.\.
							|(?:[.,;"'?():-_`])	
						"""
	text = clean_str(text)
	result = nltk.regexp_tokenize(text, pattern)
	result = [i for i in result if i not in stopworddic]

	return result


def my_ast2vec(project):
	# 将ast转换为vector
	logger = get_logger('ast word embedding', main_dir + project + '/' + project + '_ast2vec.log')
	logger.info('word embedding for code ast starting......')

	input_file = main_dir + project + '/' + project + '_ast.json'
	add_vocab_file = '../models/' + project + '_add_vocab.json'
	model_file = '../models/enwiki_20180420_100d.txt.bz2'

	logger.info('load project asts from file......')
	with open(input_file, 'r') as f:
		project_ast_list = json.load(f)

	logger.info('load add vocab from file......')
	with open(add_vocab_file, 'r') as load_f:
		add_vocab = json.load(load_f)

	logger.info('load enwiki model from file......')
	model = KeyedVectors.load_word2vec_format(model_file, binary=False)

	project_ast_vec = []
	tmp = 0

	for project_ast in project_ast_list:
		file_ast = list(tokenize_and_stopwords(project_ast['file_ast'].strip()))
		index = project_ast['file_id']

		file_ast_vec = []

		logger.info(str(index + 1) + ' / ' + str(total_files[project]))

		for word in file_ast:
			if word in model.vocab:
				file_ast_vec.append(list(np.array(model[word])))
			elif word in add_vocab:
				file_ast_vec.append(list(add_vocab[word]))
			else:
				tmp += 1
				logger.info('add vocab:      ' + word)

		project_ast_vec.append({'file_id': index, 'file_ast': file_ast_vec})

	return project_ast_vec


def save_vec(project, project_ast_vec):
	output_file = main_dir + project + '/' + project + '_ast_vec.json'
	with open(output_file, 'w') as f:
		json.dump(project_ast_vec, f, cls=MyEncoder)


def run_main(project):
	# 进行词嵌入
	logger_main.info('word embedding for code ast......')
	project_ast_vec = my_ast2vec(project)

	# 存储
	save_vec(project, project_ast_vec)

	return


if __name__ == '__main__':
	logger_main = get_logger('run_main', main_dir + project_name + '/' + 'ast_main.log')
	logger_main.info('ast process starting.............')

	run_main(project_name)

	logger_main.info('ast process ended successfully ^_^')