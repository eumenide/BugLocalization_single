#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import pandas as pd
import logging
import json
import numpy as np
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from keras.utils import Sequence
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, concatenate, Dropout, Reshape
from keras.preprocessing import sequence

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#
# # 进行配置，使用20%的GPU
# gpu_config = tf.ConfigProto()
# gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.1
# session = tf.Session(config=gpu_config)
#
# KTF.set_session(session)

main_dir = '../datasets/'
project = 'aspectj'

MAX_LEN = 200
MAX_DIM = 100
BATCH_SIZE = 50
epochs = 50

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
	# 读取数据并padding
	logger_main.info('load train data from file......')
	train_file = main_dir + project + '/' + project + '_train.json'
	train_data = pd.read_json(train_file, orient='records', dtype=False)

	logger_main.info('padding train data......')
	train_x = sequence.pad_sequences(train_data['desc'], maxlen=MAX_LEN, dtype='float32', padding='post',
	                                 truncating='post', value=0.)
	# train_y = train_data['label']
	train_y = sequence.pad_sequences(train_data['label'], maxlen=1405, dtype='int32', padding='post', truncating='post', value=0)

	logger_main.info('load test data from file......')
	test_file = main_dir + project + '/' + project + '_test.json'
	test_data = pd.read_json(test_file, orient='records', dtype=False)

	logger_main.info('padding test data......')
	test_x = sequence.pad_sequences(test_data['desc'], maxlen=MAX_LEN, dtype='float32', padding='post',
	                                truncating='post', value=0.)
	# test_y = test_data['label']
	test_y = sequence.pad_sequences(test_data['label'], maxlen=1405, dtype='int32', padding='post', truncating='post', value=0)

	logger_main.info('load validation data from file......')
	valid_file = main_dir + project + '/' + project + '_valid.json'
	valid_data = pd.read_json(valid_file, orient='records', dtype=False)

	logger_main.info('padding validation data......')
	valid_x = sequence.pad_sequences(valid_data['desc'], maxlen=MAX_LEN, dtype='float32', padding='post',
	                                truncating='post', value=0.)
	# valid_y = valid_data['label']
	valid_y = sequence.pad_sequences(valid_data['label'], maxlen=1405, dtype='int32', padding='post', truncating='post', value=0)

	return (train_x, train_y), (test_x, test_y), (valid_x, valid_y)


def train_model():
	# 定义训练模型
	# 定义Inputs
	inputs = Input(shape=[MAX_LEN, MAX_DIM], name='x_input')

	# 卷积层 三个filter_size
	convs = []
	filter_sizes = [2, 3, 4]
	for fzs in filter_sizes:
		conv_1 = Conv1D(filters=100, kernel_size=fzs, activation='relu', padding='valid')(inputs)
		pool_1 = MaxPooling1D((MAX_LEN - fzs + 1), strides=1, padding='valid')(conv_1)
		# flat_1 = Flatten()(pool_1)
		# flat_1 = Reshape((100,))(pool_1)
		convs.append(pool_1)

	# 将三个分支通过concatenate的方式拼接在一起
	merge = concatenate(convs, axis=1, name='merge')
	merge = Reshape((300,))(merge)

	# 定义dropout层，值为0.5
	dropout = Dropout(0.5)(merge)

	# 全连接层
	full_con = Dense(32, activation='relu')(dropout)

	# 输出层
	output = Dense(units=1405, activation='sigmoid')(full_con)

	model = Model([inputs], output)

	model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

	return model


def predict_result(model, testX, testY):
	# 预测结果保存
	predictY = model.predict(testX)

	predict_res = {'predict:': list(predictY), 'real:': list(testY)}

	predict_file = main_dir + project + '/' + project + '_predict_v11.json'
	with open(predict_file, 'w') as f:
		json.dump(predict_res, f, cls=MyEncoder)

	return


def run_main():
	# 主函数
	logger_main.info('load data from file......')
	(trainX, trainY), (testX, testY), (validX, validY) = load_data()

	logger_main.info('train model for data......')
	model = train_model()

	logger_main.info('fit in model......')
	model.fit(trainX, trainY,
	          validation_data=(validX, validY),
	          batch_size=BATCH_SIZE,
	          epochs=epochs,
	          shuffle=False)

	logger_main.info('save model......')
	model.save("../models/aspectj_model_v11.h5")

	logger_main.info('predict for test data......')
	predict_result(model, testX, testY)

	logger_main.info('evaluate for model......')
	scores = model.evaluate(testX, testY)
	logger_main.info('test_loss: %f, accuracy: %f' % (scores[0], scores[1]))
	print('test_loss: %f, accuracy: %f' % (scores[0], scores[1]))

	return


if __name__ == '__main__':

	logger_main = get_logger('run_main', main_dir + project + '/' + 'train_v1.0.log')
	logger_main.info('train model v1.0 starting.............')

	run_main()

	logger_main.info('train model v1.0 ended successfully ^_^')
