#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


matplotlib.rcParams['font.sans-serif'] = ['SimHei']

if __name__ == '__main__':
	project_report_file = '../datasets/aspectj/aspectj_ast_vec.json'
	# project_report_file = '../datasets/aspectj/aspectj_code_vec.json'
	# project_report_file = '../datasets/aspectj/aspectj_train.json'
	project_report = pd.read_json(project_report_file, orient='records', dtype=False)

	project_report['length'] = project_report['file_ast'].map(lambda x: len(x))
	# project_report['length'] = project_report['file_vec'].map(lambda x: len(x))
	# project_report['length'] = project_report['desc'].map(lambda x: len(x))

	num, bins, patches = plt.hist(project_report['length'], bins=201, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)

	print(num)
	print(bins)

	print(str(max(project_report['length'])))
	print(str(min(project_report['length'])))
	print(str(sum(project_report['length']) / len(project_report['length'])))
	print(str(len(project_report)))
	print(len(project_report[project_report['length'] <= 200]) / len(project_report['length']))
	print(len(project_report[project_report['length'] <= 300]) / len(project_report['length']))
	print(len(project_report[project_report['length'] <= 400]) / len(project_report['length']))
	print(len(project_report[project_report['length'] <= 500]) / len(project_report['length']))
	print(len(project_report[project_report['length'] <= 600]) / len(project_report['length']))
	print(len(project_report[project_report['length'] <= 700]) / len(project_report['length']))
	print(len(project_report[project_report['length'] <= 800]) / len(project_report['length']))
	print(len(project_report[project_report['length'] <= 900]) / len(project_report['length']))
	print(len(project_report[project_report['length'] <= 1000]) / len(project_report['length']))

	plt.xlabel("区间")
	plt.ylabel("频数/频率")
	plt.title("bug report 长度分布直方图")
	plt.grid(True)
	plt.axis([0, 1000, 0, 100])
	plt.show()

