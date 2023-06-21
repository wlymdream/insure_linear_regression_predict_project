from import_data_util.TransFileDataUtil import *
from data_processing.DataProcessingUtil import *
from train_data.TheataUtils import *

# 导入数据 拿到最原始的x和y
read_file_obj = TransFileDataUtil()
read_file_result = read_file_obj.read_file('./data/insurance.csv', 'charges')

# 将数据拆分成训练集和测试集
data_for_process = read_file_obj.get_train_and_test_data(read_file_result, 0.3, 'x_data', 'y_data')
x_train = data_for_process.get('x_train')
y_train = data_for_process.get('y_train')
x_test = data_for_process.get('x_test')
y_test = data_for_process.get('y_test')

# 进行数据的预处理
# 将数据进行归一化处理 实现x的共同富裕
data_process_obj = DataProcessingUtil()
train_monalization_data = data_process_obj.monalization_data(x_train, x_test)
x_train_monalization_data = train_monalization_data.get('x_train_monalization_data')
x_test_monalization_data = train_monalization_data.get('x_test_monalization_data')

# 将数据进行升维处理
poly_data = data_process_obj.upgrade_degree(x_train_monalization_data, x_test_monalization_data, 2, True)
x_ploy_train =  poly_data.get('x_poly_data')
x_poly_test = poly_data.get('x_poly_data_test')
# 利用模型训练得到模型并进行预测
# 线性回归
predict_obj = TheataUtils()
linear_predict_y_train = predict_obj.get_predict_data(x_ploy_train, y_train)
linear_predict_y_test = predict_obj.get_predict_data(x_poly_test, y_test)
# 岭回归
ridge_predict_y_train = predict_obj.ridge_predict_data(x_ploy_train, y_train)
ridge_predict_y_test = predict_obj.ridge_predict_data(x_poly_test, y_test)
# 梯度提升回归
gb_reg_train = predict_obj.gradient_boosting_predict_data(x_ploy_train, y_train)
gb_reg_test = predict_obj.gradient_boosting_predict_data(x_poly_test, y_test)
# 模型评估
linear_reg = predict_obj.get_estimate(y_train, linear_predict_y_train, y_test, linear_predict_y_test)
print(linear_reg)

ridge_reg = predict_obj.get_estimate(y_train, ridge_predict_y_train, y_test, ridge_predict_y_test)
print(ridge_reg)

gd_reg = predict_obj.get_estimate(y_train, gb_reg_train, y_test, gb_reg_test)
print(gd_reg)
# 发现梯度提升的这个是最好的，因为他们的



