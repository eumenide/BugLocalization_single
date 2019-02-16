#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import re
import logging
import json
import nltk
import math

from collections import Counter

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

total_files = {'aspectj': 1406, 'eclipseUI': 15179, 'jdt': 12682, 'swt': 8119, 'tomcat': 2355}

main_dir = '../datasets/'
project_name = 'aspectj'

logger_main = None

# 状态
S_INIT = 0
S_SLASH = 1
S_BLOCK_COMMENT = 2
S_BLOCK_COMMENT_DOT = 3
S_LINE_COMMENT = 4
S_STR = 5
S_STR_ESCAPE = 6


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


def automata_trim(lines):
	state = S_INIT
	ret = []
	for line in lines:
		for c in line:
			if state == S_INIT:
				if c == '/':
					state = S_SLASH
				elif c == '"':
					state = S_STR
					ret.append(c)
				else:
					ret.append(c)
			elif state == S_SLASH:
				if c == '*':
					state = S_BLOCK_COMMENT
				elif c == '/':
					state = S_LINE_COMMENT
				else:
					#  if c=='\'':
					ret.append('/')
					ret.append(c)
					state = S_INIT
			elif state == S_BLOCK_COMMENT:
				if c == '*':
					state = S_BLOCK_COMMENT_DOT
			elif state == S_BLOCK_COMMENT_DOT:
				if c == '/':
					state = S_INIT
				else:
					state = S_BLOCK_COMMENT
			elif state == S_LINE_COMMENT:
				if c == '\n':
					state = S_INIT
					ret.append(c)
			elif state == S_STR:
				if c == '\\':
					state = S_STR_ESCAPE
				elif c == '"':
					state = S_INIT
				ret.append(c)
			elif state == S_STR_ESCAPE:
				state = S_STR
				ret.append(c)
	return "".join(ret)


def my_trim(lines):
	new_lines = []
	block_patterns = ["import", "package"]
	is_in_comment = False
	for each in lines:
		raw = each
		is_blocked = False
		if each.strip().startswith("/*") or each.strip().startswith("/ *"):
			is_in_comment = True
			continue
		if each.strip().endswith("*/") or each.strip().endswith("* /"):
			is_in_comment = False
			continue
		for each_pattern in block_patterns:
			if each.strip().startswith(each_pattern):
				is_blocked = True
				break
		if (not is_blocked) and (not is_in_comment):
			raw = raw.replace("\'", "\"")
			new_lines.append(raw)
	return new_lines


def trim_file(input_name):
	with open(input_name, 'r', encoding='utf-8', errors='ignore') as f:
		lines = f.readlines()
	res = my_trim(lines)
	res = automata_trim(res)

	return res


file_list = []


def get_file(file_path):
	file_names = os.listdir(file_path)
	for file in file_names:
		new_dir = file_path + '/' + file
		if os.path.isfile(new_dir):
			add_file = re.sub(r'(../datasets/)([a-zA-Z0-9_.])*/(sourceFile/)', '', new_dir)
			file_list.append(os.path.splitext(add_file)[0])
		else:
			get_file(new_dir)


def generate_file_list(project):
	get_file(main_dir + project + '/sourceFile')
	with open(main_dir + project + '/' + project + '.txt', 'w') as f:
		for file in file_list:
			f.write(file + '\n')
	with open(main_dir + project + '/count.txt', 'w') as f:
		f.write(str(len(file_list)))


def tf_calculate(project):
	# 统计词频
	logger = get_logger('tf calculate for code', main_dir + project + '/tf_calculate.log')
	logger.info('tf calculating start......')
	input_dir = main_dir + project + '/sourceFile/'
	output_dir = main_dir + project + '/sourceFile_pre'

	temp = 1
	term_counts = []
	total_num = len(file_list)

	# 1 处理项目下的每一个文件
	for file in file_list:
		input_file = input_dir + file + '.java'
		output_file = output_dir + file + '.txt'

		logger.info(str(temp) + ' / ' + str(total_num))
		# 2 对文件进行trim操作，删掉文件中所有注释
		logger.info('trim for file......')
		file_lines = trim_file(input_file)

		counters = Counter()
		# 3~4 对文件每一行进行分词、去除停用词；并计算每个term的词频
		# 5 将新文件存储在sourceFile_pre项目下，后缀名为.txt
		logger.info('tokenize and stopwords for code.......')
		dirs = os.path.dirname(output_file)
		if not os.path.exists(dirs):
			os.makedirs(dirs)
		with open(output_file, 'w') as fout:
			for line in file_lines:
				line = line.strip()
				line = tokenize_and_stopwords(line)
				if not len(line):
					continue
				fout.write(' '.join(list(line)))
				fout.write('\n')
				counters += Counter(line)

		term_counts.append(counters)

		temp += 1

	# 8 将词频统计存储为_tf.json文件
	tf_file = main_dir + project + '/' + project + '_tf.json'
	with open(tf_file, 'w') as f:
		json.dump(term_counts, f)

	return term_counts


def generate_idf(word, tf_list):
	n_doc = sum(1 for tf in tf_list if word in tf)

	return math.log(len(tf_list) / n_doc + 0.01)


def generate_tfidf(project, term_counts):
	# todo 计算每个term的TF-IDF权重
	logger = get_logger('tf-idf calculate for term', main_dir + project + '/weight_calculate.log')
	logger.info('tf-idf calculating start......')

	weight_list = []
	temp = 1
	idf_dict = {}

	for terms in term_counts:
		logger.info(str(temp) + ' / ' + str(total_files[project]))

		weight_item = {}
		term_size = sum(terms.values())

		for term in terms:
			tf = math.log(terms[term] / term_size + 1)
			if term in idf_dict:
				idf = idf_dict[term]
			else:
				idf = generate_idf(term, term_counts)
				idf_dict[term] = idf
			weight_item[term] = tf * idf
		weight_list.append(weight_item)

		temp += 1

	weight_file = main_dir + project + '/' + project + '_w.json'
	with open(weight_file, 'w') as f:
		json.dump(weight_list, f)

	return weight_list


def code_word2vec(weight_list):
	# todo 对code进行行平均词嵌入

	return


def run_main(project):
	# 0 统计文件列表信息
	logger_main.info('generate file list......')
	generate_file_list(project)

	# 1~6 预处理并计算词频
	logger_main.info('calculate tf for code......')
	term_counts = tf_calculate(project)

	# 9 计算tf-idf权重
	logger_main.info('calculate tf-idf weight for term.......')
	weight_list = generate_tfidf(project, term_counts)

	# 12 词嵌入
	logger_main.info('word2vec for code......')
	code_word2vec(weight_list)

	# 15 将词嵌入结果存储为_code_vec.json
	logger_main.info('save code vectors to file......')


	return


if __name__ == '__main__':
	logger_main = get_logger('run_main', main_dir + project_name + '/' + 'code_main.log')
	logger_main.info('code process starting.............')

	run_main(project_name)

	logger_main.info('code process ended successfully ^_^')
