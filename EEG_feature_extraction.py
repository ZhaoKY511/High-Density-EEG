#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats
import scipy
import scipy.signal
import scipy.fftpack
import math
from numpy import mean, sqrt, square




def matrix_from_csv_file(file_path):
    csv_data = np.genfromtxt(file_path, delimiter=',')
    full_matrix = csv_data[1:]

    return full_matrix


# 时间切片
def get_time_slice(full_matrix, start=0., period=1.0):
    rstart = full_matrix[0, 0] + start
    index_0 = np.max(np.where(full_matrix[:, 0] <= rstart))
    index_1 = np.max(np.where(full_matrix[:, 0] <= rstart + period))

    duration = full_matrix[index_1, 0] - full_matrix[index_0, 0]
    return full_matrix[index_0:index_1, :], duration


# 整数秒时间窗口内每个窗口内的每个信号的平均值（我们的输入有四列值，四个频段求了平均）
def feature_mean(matrix: object):
    """
	返回整个时间窗口内每个信号的平均值

	Parameters:
		单个时间窗口matrix (numpy.ndarray): 2D [nsamples x nsignals]矩阵，其中包含长度为nsamples的时间窗口的nsignals值

	Returns:
		numpy.ndarray: 一维数组，包含输入矩阵中每一列的均值
		list: 包含计算出的数量的特征名称的列表。

	"""

    ret = np.mean(matrix, axis=0).flatten()#axis=0表示按列求平均，输出为一行包含每一列的平均值
    names = ['mean_' + str(i) for i in range(matrix.shape[1])]
    return ret, names





def feature_stddev(matrix):
    """

	在整个时间范围内计算每个信号的标准差

	Parameters:

		2D [nsamples x nsignals]矩阵，其中包含长度为nsamples的时间窗口的nsignals值

	Returns:
		numpy.ndarray: 一维数组，其中包含每列与输入矩阵的标准偏差


	"""

    #fix ddof for finite sampling correction (N-1 instead of N in denominator)
    ret = np.std(matrix, axis=0, ddof=1).flatten()
    names = ['std_' + str(i) for i in range(matrix.shape[1])]

    return ret, names





def feature_max(matrix):
    """
	Returns the maximum value of each signal for the full time window
	返回整个时间窗口内每个信号的最大值
	Parameters:
		matrix (numpy.ndarray): 2D [nsamples x nsignals] matrix containing the
		values of nsignals for a time window of length nsamples

	Returns:
		numpy.ndarray: 1D array containing the max of each column from the input matrix
		list: list containing feature names for the quantities calculated.

	"""

    ret = np.max(matrix, axis=0).flatten()
    names = ['max_' + str(i) for i in range(matrix.shape[1])]
    return ret, names





def feature_min(matrix):
    """

	返回整个时间范围内每个信号的最小值

		matrix (numpy.ndarray): 2D [nsamples x nsignals]矩阵，其中包含长度为nsamples的时间窗口的nsignals值

	"""

    ret = np.min(matrix, axis=0).flatten()
    names = ['min_' + str(i) for i in range(matrix.shape[1])]
    return ret, names


def feature_skew(matrix):


    ret = stats.skew(matrix, axis=0)  #axis=0表示按列求平均，输出为一行包含每一列的平均值
    names = ['skew_' + str(i) for i in range(matrix.shape[1])]

    return ret, names

def feature_kurt(matrix):


    ret = stats.skew(matrix, axis=0).flatten()#axis=0表示按列求平均，输出为一行包含每一列的平均值
    names = ['kurt_' + str(i) for i in range(matrix.shape[1])]
    return ret, names

def feature_rms(matrix):
    ret = np.sqrt(np.mean(np.square(matrix), axis=0))

    names = ['rms_' + str(i) for i in range(matrix.shape[1])]
    return ret, names

def feature_differential_entropy(matrix):

    # 差分熵公式：h(X) = 0.5 * log(2 * π * e * variance)
    variances = np.var(matrix, axis=0)  # 按列计算方差
    ret = 0.5 * np.log(2 * np.pi * np.e * variances)
    names = ['DE_' + str(i) for i in range(matrix.shape[1])]
    return ret, names
"""
"添加了新的统计学特征偏度和峰度"
def feature_skew(matrix):


    ret = stats.skew(matrix, axis=0).flatten()#axis=0表示按列求平均，输出为一行包含每一列的平均值
    names = ['skew_' + str(i) for i in range(matrix.shape[1])]

    return ret, names

def feature_kurt(matrix):


    ret = stats.skew(matrix, axis=0).flatten()#axis=0表示按列求平均，输出为一行包含每一列的平均值
    names = ['kurt_' + str(i) for i in range(matrix.shape[1])]
    return ret, names

def feature_rms(matrix):
    ret = math.sqrt(np.mean(square(matrix)))

    names = ['rms_' + str(i) for i in range(matrix.shape[1])]
    return ret, names

def feature_hjorth(matrix):
    ret = np.square(np.std(matrix))

    names = ['hjorth_' + str(i) for i in range(matrix.shape[1])]
    return ret, names

def feature_differential_entropy(matrix):

    # 差分熵公式：h(X) = 0.5 * log(2 * π * e * variance)
    ret = 0.5 * np.log(2 * np.pi * np.e *  np.var(matrix))
    names = ['DE_' + str(i) for i in range(matrix.shape[1])]
    return ret, names



def feature_covariance_matrix(matrix):

    Computes the elements of the covariance matrix of the signals. Since the
    covariance matrix is symmetric, only the lower triangular elements
    (including the main diagonal elements, i.e., the variances of eash signal)
    are returned.

    Parameters:
        matrix (numpy.ndarray): 2D [nsamples x nsignals] matrix containing the
        values of nsignals for a time window of length nsamples

    Returns:
        numpy.ndarray: 1D array containing the variances and covariances of the
        signals
        list: list containing feature names for the quantities calculated.
        numpy.ndarray: 2D array containing the actual covariance matrix
    Author:
        Original: [fcampelo]
 

    covM = np.cov(matrix.T)
    indx = np.triu_indices(covM.shape[0])
    ret = covM[indx]

    names = []
    for i in np.arange(0, covM.shape[1]):
        for j in np.arange(i, covM.shape[1]):
            names.extend(['covM_' + str(i) + '_' + str(j)])

    return ret, names
"""


def calc_feature_vector(matrix, state):
    """
	计算所有先前定义的特征，并将所有内容合并为一个特征向量。

	"""
#    h1, h2 = np.split(matrix, [int(matrix.shape[0] / 2)])
#    q1, q2, q3, q4 = np.split(matrix,
#                              [int(0.25 * matrix.shape[0]),
#                               int(0.50 * matrix.shape[0]),
 #                              int(0.75 * matrix.shape[0])])


    var_names = []

    x, v = feature_mean(matrix)
    var_names += v
    var_values = x

    x, v = feature_stddev(matrix)
    var_names += v
    var_values = np.hstack([var_values, x])

    x, v = feature_max(matrix)
    var_names += v
    var_values = np.hstack([var_values, x])

    x, v = feature_min(matrix)
    var_names += v
    var_values = np.hstack([var_values, x])

    x, v = feature_skew(matrix)
    var_names += v
    var_values = np.hstack([var_values, x])

    x, v = feature_kurt(matrix)
    var_names += v
    var_values = np.hstack([var_values, x])

    x, v = feature_rms(matrix)
    var_names += v
    var_values = np.hstack([var_values, x])

    x, v = feature_differential_entropy(matrix)
    var_names += v
    var_values = np.hstack([var_values, x])









    if state != None:
        var_values = np.hstack([var_values, np.array([state])])
        var_names += ['Label']

    return var_values, var_names


def generate_feature_vectors_from_samples(file_path, nsamples, period,
                                          state=None,
                                          remove_redundant=True,
                                          cols_to_ignore=None):
    # 读取文件
    matrix = matrix_from_csv_file(file_path)

    # 从第一个文件开始
    t = 0.

    previous_vector = None

    ret = None

    # Until an exception is raised or a stop condition is met
    while True:
        # 't'开始, 'period'为1s时间窗
        # 如果引发异常或切片时间异常，返回当前可用数据
        try:
            s, dur = get_time_slice(matrix, start=t, period=period)
            if cols_to_ignore is not None:
                s = np.delete(s, cols_to_ignore, axis=1)
        except IndexError:
            break
        if len(s) == 0:
            break
        if dur < 0.9 * period:
            break

        ry, rx = scipy.signal.resample(s[:, 1:], num=nsamples,
                                       t=s[:, 0], axis=0)

        # 重叠50%
        t += 1 * period

        # 计算特征向量。 我们将附加当前时间片的特征和前一个时间片的特征，如果没有先前的向量，则将其设置并继续下一个向量。
        r, headers = calc_feature_vector(ry, state)

        if previous_vector is not None:

            feature_vector = np.hstack([previous_vector, r])

            if ret is None:
                ret = feature_vector
            else:
                ret = np.vstack([ret, feature_vector])

        # 保存之前的窗口
        previous_vector = r
        if state is not None:
            # 删掉之前的vector
            previous_vector = previous_vector[:-1]

    feat_names = ["lag1_" + s for s in headers[:-1]] + headers



    return ret, feat_names

#
