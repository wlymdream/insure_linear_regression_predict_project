# 数据预处理类
import numpy as np
# 做数据处理 利用归一处理
from sklearn.preprocessing import StandardScaler
# 正则化，惩罚项
from sklearn.preprocessing import PolynomialFeatures
from import_data_util.TransFileDataUtil import *

'''
    将数据进行归一化处理，为了统计theata,进行归一化处理的原因是因为梯度下降法的时候，
    我们的不同维度的x下降的频次它是不一样的，我们尽量的把它实现共同富裕，让x的改变频次尽量的变一样  
'''

class DataProcessingUtil:

    '''
        将数据进行归一化处理：实现x的共同富裕，在做梯度下降法的时算theata候会很有帮助
    '''
    def monalization_data(_self, x_train, x_test):
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(x_train)
        x_train_scale = scaler.transform(x_train)
        x_test_scale = scaler.transform(x_test)
        result = {'x_train_monalization_data': x_train_scale, 'x_test_monalization_data': x_test_scale}
        return result

    '''
        利用升维更精确化theata， 提高欠拟合，提高模型的准确率，
        degree_size 多少阶
        include_bias_status 考不考虑截距项 y =theata0 + x1 theata1 + ……
    '''
    def upgrade_degree(_self, x_train_monalizatioin, x_test_monalization, degree_size, include_bias_status):
        polynomial_obj = PolynomialFeatures(degree=degree_size, include_bias=include_bias_status)
        x_train = polynomial_obj.fit_transform(x_train_monalizatioin)
        x_test = polynomial_obj.fit_transform(x_test_monalization)
        return {'x_poly_data' : x_train, 'x_poly_data_test' : x_test}

