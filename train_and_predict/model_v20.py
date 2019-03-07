#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import json
import pandas as pd
import logging
import numpy as np

from keras.models import Model
from keras.utils import Sequence, to_categorical
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, concatenate, Dropout, Reshape, Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.preprocessing import sequence

main_dir = '../datasets/'
project = 'aspectj'

# MAX_LEN = 200
MAX_LEN = 500
MAX_DIM = 100
BATCH_SIZE = 64
epochs = 20

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


class DataGenerator(Sequence):
	'Generate data for keras'

	def __init__(self, reports, codes, labels=None, batch_size=64, is_y=True, name=''):
		'Initialization'
		self.reports = reports
		self.codes = codes
		self.is_y = is_y
		self.name = name
		if self.is_y:
			self.labels = labels
			self.label_index = 0
			self.label_len = len(self.labels)
		self.batch_size = batch_size
		self.report_index = 0
		self.code_index = 0
		self.code_len = len(self.codes)
		self.report_len = len(self.reports)
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.ceil(self.code_len / self.batch_size) * self.report_len)

	def on_epoch_end(self):
		'Update indexs after each epoch'
		self.report_index = 0
		self.code_index = 0
		if self.is_y:
			self.label_index = 0

	def __getitem__(self, item):
		'Generate one batch of data'
		# 计算第item次batch的code、report以及label的初始index
		self.code_index = int(np.floor(item * self.batch_size % self.code_len))
		self.report_index = int(np.floor(item * self.batch_size / self.code_len))
		if self.is_y:
			self.label_index = self.report_index

		# 根据index获取数据集
		report_tmp = self.reports[self.report_index]
		code_tmp = self.codes[self.code_index: self.code_index + self.batch_size]

		# 将report数据复制batch_size份，并同时将label转换成二分类格式
		reports = report_tmp[np.newaxis, :]
		for i in range(len(code_tmp) - 1):
			reports = np.vstack((reports, report_tmp[np.newaxis, :]))

		if self.is_y:
			label_tmp = self.labels[self.label_index]
			label = label_tmp[self.code_index * self.batch_size: (self.code_index + 1) * self.batch_size]
			label = to_categorical(label, num_classes=2)
			return [reports, code_tmp], label
		else:
			return [reports, code_tmp]


def load_data():
	# 读取数据并padding
	logger_main.info('load train data from file......')
	train_file = main_dir + project + '/' + project + '_train.json'
	train_data = pd.read_json(train_file, orient='records', dtype=False)

	logger_main.info('padding train data......')
	train_x = sequence.pad_sequences(train_data['desc'], maxlen=MAX_LEN, dtype='float32', padding='post',
	                                 truncating='post', value=0.)
	# train_y = train_data['label']
	train_y = sequence.pad_sequences(train_data['label'], maxlen=1405, dtype='int32', padding='post', truncating='post',
	                                 value=0)

	logger_main.info('load test data from file......')
	test_file = main_dir + project + '/' + project + '_test.json'
	test_data = pd.read_json(test_file, orient='records', dtype=False)

	logger_main.info('padding test data......')
	test_x = sequence.pad_sequences(test_data['desc'], maxlen=MAX_LEN, dtype='float32', padding='post',
	                                truncating='post', value=0.)
	# test_y = test_data['label']
	test_y = sequence.pad_sequences(test_data['label'], maxlen=1405, dtype='int32', padding='post', truncating='post',
	                                value=0)

	logger_main.info('load validation data from file......')
	valid_file = main_dir + project + '/' + project + '_valid.json'
	valid_data = pd.read_json(valid_file, orient='records', dtype=False)

	logger_main.info('padding validation data......')
	valid_x = sequence.pad_sequences(valid_data['desc'], maxlen=MAX_LEN, dtype='float32', padding='post',
	                                 truncating='post', value=0.)
	# valid_y = valid_data['label']
	valid_y = sequence.pad_sequences(valid_data['label'], maxlen=1405, dtype='int32', padding='post', truncating='post',
	                                 value=0)

	logger_main.info('load source code vectors from file......')
	code_file = main_dir + project + '/' + project + '_code_vec.json'
	code_data = pd.read_json(code_file, orient='records', dtype=False)

	logger_main.info('padding code vectors......')
	code_input = sequence.pad_sequences(code_data['file_vec'], maxlen=MAX_LEN, dtype='float32', padding='post',
	                                    truncating='post', value=0)

	return (train_x, train_y), (test_x, test_y), (valid_x, valid_y), code_input


def train_model():
	#
	# 定义Inputs
	report_input = Input(shape=[MAX_LEN, MAX_DIM], name='report')
	code_input = Input(shape=[MAX_LEN, MAX_DIM], name='code')

	# bug report卷积层 三个filter_size
	convs = []
	filter_sizes = [2, 3, 4]
	for fzs in filter_sizes:
		conv_1 = Conv1D(filters=100, kernel_size=fzs, activation='relu', padding='valid')(report_input)
		pool_1 = MaxPooling1D((MAX_LEN - fzs + 1), strides=1, padding='valid')(conv_1)
		convs.append(pool_1)

	# 将bug report三个分支通过concatenate的方式拼接在一起
	merge_report = concatenate(convs, axis=1)

	# flatten层
	flatten_report = Reshape((1, 300))(merge_report)

	# source code卷积层 三个filter_size
	convs = []
	filter_sizes = [2, 3, 4]
	for fzs in filter_sizes:
		conv_1 = Conv1D(filters=100, kernel_size=fzs, activation='relu', padding='valid')(code_input)
		pool_1 = MaxPooling1D((MAX_LEN - fzs + 1), strides=1, padding='valid')(conv_1)
		convs.append(pool_1)

	# 将source code三个分支通过concatenate的方式拼接在一起
	merge_code = concatenate(convs, axis=1)

	# flatten层
	flatten_code = Reshape((1, 300))(merge_code)

	# 将相同维度的report code和source code特征进行连接
	merge_all = concatenate([flatten_report, flatten_code], axis=1)

	print(merge_all.shape)
	flatten_all = Reshape((2, 300, 1))(merge_all)
	print(flatten_all.shape)

	# 以下为CNN分类器
	# 卷积层
	conv_classify = Conv2D(filters=100, kernel_size=(2, 2), activation='relu', padding='same')(flatten_all)
	print(conv_classify.shape)

	# 池化层
	pool_classify = MaxPooling2D((2, 300), strides=1, padding='valid')(conv_classify)

	# flatten_classify = Reshape((100, ))(pool_classify)
	flatten_classify = Flatten()(pool_classify)
	# 定义dropout层，值为0.5
	dropout = Dropout(0.5)(flatten_classify)

	# 全连接层
	full_con = Dense(32, activation='relu')(dropout)

	output = Dense(2, activation='sigmoid')(full_con)

	model = Model([report_input, code_input], output)

	model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

	print(model.summary())
	return model


def predict_result(model, test_generator, testY):
	# todo
	# 预测结果保存
	predictY = model.predict_generator(generator=test_generator)

	predict_res = {'predict:': list(predictY), 'real:': list(to_categorical(testY))}

	predict_file = main_dir + project + '/' + project + '_predict_v11.json'
	with open(predict_file, 'w') as f:
		json.dump(predict_res, f, cls=MyEncoder)

	return


def run_main():
	# 主函数
	logger_main.info('load data from file......')
	(trainX, trainY), (testX, testY), (validX, validY), code_input = load_data()

	logger_main.info('train model for data......')
	model = train_model()

	logger_main.info('fit in model......')
	train_generator = DataGenerator(trainX, code_input, labels=trainY, name='train')
	valid_generator = DataGenerator(validX, code_input, labels=validY, name='valid')
	test_generator = DataGenerator(testX, code_input, is_y=False, name='test')
	eval_generator = DataGenerator(testX, code_input, labels=testY, name='eval')

	model.fit_generator(generator=train_generator,
	                    validation_data=valid_generator,
	                    epochs=epochs,
	                    shuffle=False)
	# model.fit(trainX, trainY,
	#           validation_data=(validX, validY),
	#           batch_size=BATCH_SIZE,
	#           epochs=epochs,
	#           shuffle=False)

	logger_main.info('save model......')
	model.save("../models/aspectj_model_v20.h5")

	logger_main.info('predict for test data......')
	predict_result(model, test_generator, testY)

	logger_main.info('evaluate for model......')
	scores = model.evaluate_generator(eval_generator)
	logger_main.info('test_loss: %f, accuracy: %f' % (scores[0], scores[1]))
	print('test_loss: %f, accuracy: %f' % (scores[0], scores[1]))

	return


if __name__ == '__main__':
	logger_main = get_logger('run_main', main_dir + project + '/' + 'train_v20.log')
	logger_main.info('train model v2.0 starting.............')

	run_main()

	logger_main.info('train model v2.0 ended successfully ^_^')
