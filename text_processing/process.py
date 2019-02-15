#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import re
import pandas as pd
import nltk
import numpy as np
import json
import logging

from nltk.corpus import stopwords
from gensim.models import KeyedVectors


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

total_files = {'aspectj': 1406, 'eclipseUI': 15179, 'jdt': 12682, 'swt': 8119, 'tomcat': 2355}

logger_main = None

stopwords_dic = set(stopwords.words('english')) | {'.', ':', '(', ')', '\n'}
main_dir = '../datasets/'
project_name = 'aspectj'

model = None

with open('../models/add_vocab.json', 'r') as load_f:
	add_vocab = json.load(load_f)


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


def load_data_from_xsl(file_name):
	nlp_data = pd.read_excel(file_name, sheet_name=0, header=0, usecols=[1, 2, 3],
	                         converters={'bug_id': str, 'summary': str, 'description': str}, nrows=3)
	nlp_data.fillna(' ', inplace=True)

	return nlp_data


def preprocess(data):
	data['summary'] = data['summary'].map(lambda x: x[x.find(' ', 4):])
	data['desc'] = data.apply(lambda x: x['summary'] + ' ' + x['description'], axis=1)

	data.drop(labels=['summary', 'description'], axis=1, inplace=True)

	data['desc'] = data['desc'].map(lambda x: tokenize_and_stopwords(x))

	return data


temp = 1


def my_embedding(data, logger):
	global temp
	logger.info(project_name + '    ' + str(temp) + ' / ' + total_files[project_name])
	vec = []
	for string in data:
		if string in model.vocab:
			vec.append(list(model[string]))
		elif string in add_vocab:
			vec.append(list(add_vocab[string]))
		else:
			ran = list(np.random.uniform(-1, 1, 100))
			vec.append(ran)
			add_vocab[string] = ran
			logger.info('add_vocab    ' + string)

	temp = temp + 1
	return vec


def my_word2vec(data):
	logger = get_logger('word embedding', main_dir + project_name + '/' + 'word2vec.log')
	data = data.map(lambda x: my_embedding(x, logger))

	logger.info('save add_vocab......')
	with open('../models/add_vocab.json', 'w') as f:
		json.dump(add_vocab, f)

	return data


def generate_feature(data):
	logger = get_logger('extract feature', main_dir + project_name + '/' + 'extract_feature.log')

	# todo


	return data


def run_main():
	logger_main.info('load enwiki model......')
	global model
	model = KeyedVectors.load_word2vec_format('../models/enwiki_20180420_100d.txt.bz2', binary=False)
	logger_main.info('load model successfully')

	xlsx_file = main_dir + project_name + '/' + project_name + '.xlsx'
	# 读取数据
	logger_main.info('load data from xsl......')
	nlp_data = load_data_from_xsl(xlsx_file)

	# 分词、去除停用词
	logger_main.info('preprocess for data......')
	nlp_data = preprocess(nlp_data)

	# 对desc字段进行词嵌入
	logger_main.info('word embedding for data.....')
	nlp_data['desc'] = my_word2vec(nlp_data['desc'])

	# 将desc字段中的词向量进行卷积操作
	logger_main.info('extract feature for data.....')
	nlp_data['desc'] = generate_feature(nlp_data['desc'])

	# 将特征提取结果存储为aspectj_report_feature.json
	logger_main.info('save feature result to file......')
	feature_file = main_dir + project_name + '/' + project_name + '_report_feature.json'
	nlp_data.to_json(feature_file, orient='records')


if __name__ == '__main__':
	logger_main = get_logger('run_main', main_dir + project_name + '/' + 'main.log')
	logger_main.info('text process starting.............')

	run_main()

	logger_main.info('text process ended successfully ^_^')



