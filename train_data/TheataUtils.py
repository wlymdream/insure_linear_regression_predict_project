# 模型训练
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
# 梯度提升
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

'''
    由于我们是使用的线性回归，那么我们假设我们的误差是服从正态分布的，我们的数据服从正态分布更合理一些，
    所以y要做一些变换 log 
    如果最后模型评估的结果不是很好的话，误差较大的话，那么它可能并不是一个线性回归的数据，我们需要换个模型来处理
    
    
'''
class TheataUtils :

    '''
        获取预测数据 不带正则项 linearRegression
    '''
    def get_predict_data(self, x, y):
        linear_reg = LinearRegression()
        # y把它变成正态分布的 log1p log的升级版
        linear_reg.fit(x, np.log1p(y))
        y_predict = linear_reg.predict(x)
        return y_predict

    '''
        带正则项的数据预测 岭回归
    '''
    def ridge_predict_data(self, x, y):
        ridge_reg = Ridge(alpha=0.4)
        ridge_reg.fit(x, np.log1p(y))
        y_predict = ridge_reg.predict(x)
        return y_predict

    '''
        利用梯度提升来提高模型准确率回归
    '''
    def gradient_boosting_predict_data(self, x, y):
        gb_reg = GradientBoostingRegressor()
        gb_reg.fit(x, np.log1p(y))
        y_predict =  gb_reg.predict(x)
        return y_predict

    '''
        模型评估  (y_hat - y )^2 我们给他开根号 如果误差比较大的话，那么这个模型他不是很好的
    '''
    def get_estimate(self, y_train, y_predicts, y_test, y_test_predict):
        # 由于我们的数据是真实的比对， 所以我们把预测的y给还原 之前的y是log的
        mse = mean_squared_error(y_train, np.exp(y_predicts))
        # 开根号
        rmse = np.sqrt(mse)

        mse_test = mean_squared_error(y_test, np.exp(y_test_predict))
        rmse_test = np.sqrt(mse_test)
        return {'train_estimate': rmse, 'test_estimate' : rmse_test}




