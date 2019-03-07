#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from keras.models import load_model

if __name__ == '__main__':
	model = load_model('../models/aspectj_model_v11.h5')
	layers = model.layers
	for layer in layers:
		print(layer)
		print(layer.input.shape)
		print(layer.output.shape)
