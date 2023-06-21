import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

class TransFileDataUtil:
    # 读取文件
    '''
    read_file :
              y_name: 作为y列的名字
              file_dictionary ： 文件路径
    '''
    def read_file(_self, file_dictionary, y_name):
        # todo 后期把参数改了
        # 导入文件 用，分隔 返回DataFrame数据类型 返回二维表，有行有列
        data = pd.read_csv(file_dictionary, sep=',')
        # 丢掉charges这一列，其余数据作为x # todo 后期把参数改了
        x_data = data.drop(y_name, axis=1)
        # 利用one-hot编码，将字符串数据修改
        x_trans_data = pd.get_dummies(x_data)
        # 截取y数据
        y = data[y_name]
        # 处理空值
        x_trans_data.fillna(0, inplace=True)
        y.fillna(0, inplace=True)
        result = {'x_data': x_trans_data, 'y_data': y}
        return result

    '''
    将数据拆分成训练接和测试集
    data_for_trans : 元组类型 
    test_trans_size: 测试集保留多少 0.3
    x_trans_key: 元组中x的key
    y_trans_key : 元组中y的key
    '''
    def get_train_and_test_data(_self, data_for_trans, test_trans_size, x_trans_key, y_trans_key):
        # 将测试集保留test_trans_size 剩下的都作为是训练集的数据
        x_train, x_test, y_train, y_test = train_test_split(data_for_trans.get(x_trans_key), data_for_trans.get(y_trans_key), test_size=test_trans_size)
        result = {'x_train': x_train, 'y_train' : y_train, 'x_test' : x_test, 'y_test' : y_test}
        return result



# a = TransFileDataUtil()
# data_csv = a.read_file("../data/insurance.csv", "charges")
# x_data =a.get_train_and_test_data(data_csv, 0.1, "", "")
# print(x_data.head(6))
'''
    我们是打算做保险花销的预测，那么根据这个数据来看，我们可以把charges这一列的值当做是y 
    一说到预测，我们一般想到的是线性回归，线性回归的从根上来说假设数据服从一个正态分布的，最终的损失函数是MSE,
    当我们把这个y用柱状图画出来之后
    会发现数据是右偏的，我们会用np.log函数进行校正，
    
    当我们拿到了这数据之后，会发现，有的列是字符串的，比如sex的值有female和male  smoker 有yes/no 所以这里就涉及到一个
    独热one-hot编码问题，它是将这一列会拆成多列，用0和1代替 
    关于one-hot可以用sklearn.preprocessing  LabelEncoder 或者是OneHotEncoder 
    上面的代码read_csv之后拿到的数据是DataFrame dataframe里面的map也可以去做这个
    pandas里面有一个get_dummies也可以做 值都变成了true/false
'''

