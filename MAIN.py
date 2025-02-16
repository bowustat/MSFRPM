import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, LSTM,Conv1D,Dropout,Bidirectional,Multiply,Concatenate,Add,Flatten,Reshape,GRU
from keras.models import Model
tf.compat.v1.enable_eager_execution()
from keras import regularizers

from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

'''from keras import regularizers

model.add(Dense(64, input_dim=64, kernel_regularizer=regularizers.l2(0.01)'''

window_size = 1
fea_num = 16
city_FEATURES_NUM = 4


def attention_model():
    inputs1 = Input((fea_num,window_size ))
    inputs2 = Input(( 1, city_FEATURES_NUM))
    inputs3 = Input((1, city_FEATURES_NUM))
    inputs4 = Input((1, city_FEATURES_NUM))
    inputs5 = Input((1, city_FEATURES_NUM))
    inputs6 = Input((1, city_FEATURES_NUM))
    inputs7 = Input((1, city_FEATURES_NUM))
    inputs8 = Input((1, city_FEATURES_NUM))
    inputs9 = Input((1, city_FEATURES_NUM))
    inputs10 = Input((1, city_FEATURES_NUM))
    inputs11 = Input((1, city_FEATURES_NUM))
    inputs12 = Input((1, city_FEATURES_NUM))
    inputs13 = Input((1, city_FEATURES_NUM))
    inputs14 = Input((1, city_FEATURES_NUM))
    inputs15 = Input((1, city_FEATURES_NUM))
    inputs16 = Input((1, city_FEATURES_NUM))
    inputs17 = Input((1, city_FEATURES_NUM))
    x1 = Conv1D(filters = 64, kernel_size = 2, strides=1, padding="same", activation = 'relu')(inputs1)
    x2 = Conv1D(filters=64, kernel_size=2, strides=1, padding="same", activation='relu')(x1)
    x3 = Conv1D(filters=64, kernel_size=2, strides=1, padding="same", activation='relu')(x2)
    x4 = Concatenate(axis=1)([x1, x2, x3])
    x5 = Flatten()(x4)
    x = Reshape((window_size, -1))(x5)
    #x = Dropout(0.1)(x)
    #lstm_out = Bidirectional(LSTM(lstm_units, activation='relu'), name='bilstm')(x)
    #对于GPU可以使用CuDNNLSTM
    cnn_lstm = LSTM(180, activation='relu', return_sequences=True)(x)
    drop2 = Dropout(0.1)(cnn_lstm)
    cnn_lstm = LSTM(90, activation='relu', return_sequences=True)(drop2)
    drop2 = Dropout(0.1)(cnn_lstm)
    inputs01 = Reshape((window_size, -1))(inputs1)
    output_Dense1 = Dense(200, activation="tanh")(inputs2)
    output_Dense1 = Dropout(0.1)(output_Dense1)
    output_Dense1 = Dense(50, activation="tanh")(output_Dense1)
    x21 = Concatenate(axis=2)([output_Dense1, drop2, inputs01])
    Dense1 = Dense(50, activation="tanh")(x21)
    city_out1 = Dense(1, activation="tanh",name="output1")(Dense1)

    output_Dense2 = Dense(200, activation="tanh")(inputs3)
    output_Dense2 = Dropout(0.1)(output_Dense2)
    output_Dense2 = Dense(50, activation="tanh")(output_Dense2)
    x22 = Concatenate(axis=2)([output_Dense2, drop2, inputs01])
    Dense2 = Dense(50, activation="tanh")(x22)
    city_out2 = Dense(1, activation="tanh",name="output2")(Dense2)

    output_Dense3 = Dense(200, activation="tanh")(inputs4)
    output_Dense3 = Dropout(0.1)(output_Dense3)
    output_Dense3 = Dense(50, activation="tanh")(output_Dense3)
    x23 = Concatenate(axis=2)([output_Dense3, drop2, inputs01])
    Dense3 = Dense(50, activation="tanh")(x23)
    city_out3 = Dense(1, activation="tanh",name="output3")(Dense3)

    output_Dense4 = Dense(200, activation="tanh")(inputs5)
    output_Dense4 = Dropout(0.1)(output_Dense4)
    output_Dense4 = Dense(50, activation="tanh")(output_Dense4)
    x24 = Concatenate(axis=2)([output_Dense4, drop2, inputs01])
    Dense4 = Dense(50, activation="tanh")(x24)
    city_out4 = Dense(1, activation="tanh",name="output4")(Dense4)

    output_Dense5 = Dense(200, activation="tanh")(inputs6)
    output_Dense5 = Dropout(0.1)(output_Dense5)
    output_Dense5 = Dense(50, activation="tanh")(output_Dense5)
    x25 = Concatenate(axis=2)([output_Dense5, drop2, inputs01])
    Dense5 = Dense(50, activation="tanh")(x25)
    city_out5 = Dense(1, activation="tanh",name="output5")(Dense5)

    output_Dense6 = Dense(200, activation="tanh")(inputs7)
    output_Dense6 = Dropout(0.1)(output_Dense6)
    output_Dense6 = Dense(50, activation="tanh")(output_Dense6)
    x26 = Concatenate(axis=2)([output_Dense6, drop2, inputs01])
    Dense6 = Dense(50, activation="tanh")(x26)
    city_out6 = Dense(1, activation="tanh",name="output6")(Dense6)

    output_Dense7 = Dense(200, activation="tanh")(inputs8)
    output_Dense7 = Dropout(0.1)(output_Dense7)
    output_Dense7 = Dense(50, activation="tanh")(output_Dense7)
    x27 = Concatenate(axis=2)([output_Dense7, drop2, inputs01])
    Dense7 = Dense(50, activation="tanh")(x27)
    city_out7 = Dense(1, activation="tanh",name="output7")(Dense7)

    output_Dense8 = Dense(200, activation="tanh")(inputs9)
    output_Dense8 = Dropout(0.1)(output_Dense8)
    output_Dense8 = Dense(50, activation="tanh")(output_Dense8)
    x28 = Concatenate(axis=2)([output_Dense8, drop2, inputs01])
    Dense8 = Dense(50, activation="tanh")(x28)
    city_out8 = Dense(1, activation="tanh",name="output8")(Dense8)

    output_Dense9 = Dense(200, activation="tanh")(inputs10)
    output_Dense9 = Dropout(0.1)(output_Dense9)
    output_Dense9 = Dense(50, activation="tanh")(output_Dense9)
    x29 = Concatenate(axis=2)([output_Dense9, drop2, inputs01])
    Dense9 = Dense(50, activation="tanh")(x29)
    city_out9 = Dense(1, activation="tanh",name="output9")(Dense9)

    output_Dense10 = Dense(200, activation="tanh")(inputs11)
    output_Dense10 = Dropout(0.1)(output_Dense10)
    output_Dense10 = Dense(50, activation="tanh")(output_Dense10)
    x210 = Concatenate(axis=2)([output_Dense10, drop2, inputs01])
    Dense10 = Dense(50, activation="tanh")(x210)
    city_out10 = Dense(1, activation="tanh",name="output10")(Dense10)

    output_Dense11 = Dense(200, activation="tanh")(inputs12)
    output_Dense11 = Dropout(0.1)(output_Dense11)
    output_Dense11 = Dense(50, activation="tanh")(output_Dense11)
    x211 = Concatenate(axis=2)([output_Dense11, drop2, inputs01])
    Dense11 = Dense(50, activation="tanh")(x211)
    city_out11 = Dense(1, activation="tanh",name="output11")(Dense11)

    output_Dense12 = Dense(200, activation="tanh")(inputs13)
    output_Dense12 = Dropout(0.1)(output_Dense12)
    output_Dense12 = Dense(50, activation="tanh")(output_Dense12)
    x212 = Concatenate(axis=2)([output_Dense12, drop2, inputs01])
    Dense12 = Dense(50, activation="tanh")(x212)
    city_out12 = Dense(1, activation="tanh",name="output12")(Dense12)

    output_Dense13 = Dense(200, activation="tanh")(inputs14)
    output_Dense13 = Dropout(0.1)(output_Dense13)
    output_Dense13 = Dense(50, activation="tanh")(output_Dense13)
    x213= Concatenate(axis=2)([output_Dense13, drop2, inputs01])
    Dense13 = Dense(50, activation="tanh")(x213)
    city_out13 = Dense(1, activation="tanh",name="output13")(Dense13)

    output_Dense14 = Dense(200, activation="tanh")(inputs15)
    output_Dense14 = Dropout(0.1)(output_Dense14)
    output_Dense14 = Dense(50, activation="tanh")(output_Dense14)
    x214 = Concatenate(axis=2)([output_Dense14, drop2, inputs01])
    Dense14 = Dense(50, activation="tanh")(x214)
    city_out14 = Dense(1, activation="tanh",name="output14")(Dense14)

    output_Dense15 = Dense(200, activation="tanh")(inputs16)
    output_Dense15 = Dropout(0.1)(output_Dense15)
    output_Dense15 = Dense(50, activation="tanh")(output_Dense15)
    x215 = Concatenate(axis=2)([output_Dense15, drop2, inputs01])
    Dense15 = Dense(50, activation="tanh")(x215)
    city_out15 = Dense(1, activation="tanh",name="output15")(Dense15)

    output_Dense16 = Dense(200, activation="tanh")(inputs17)
    output_Dense16 = Dropout(0.1)(output_Dense16)
    output_Dense16 = Dense(50, activation="tanh")(output_Dense16)
    x216 = Concatenate(axis=2)([output_Dense16, drop2, inputs01])
    Dense16 = Dense(50, activation="tanh")(x216)
    city_out16 = Dense(1, activation="tanh",name="output16")(Dense16)

    z_feature = Concatenate(axis=2)([city_out1,city_out2,city_out3,city_out4,city_out5,city_out6,city_out7,city_out8, city_out9,city_out10,city_out11,city_out12,city_out13,city_out14,city_out15,city_out16])
    output = Dense(16, activation="tanh")(z_feature)

    model = Model(inputs=[inputs1,inputs2,inputs3,inputs4,inputs5, inputs6,inputs7,inputs8, inputs9, inputs10,inputs11,inputs12,inputs13, inputs14,inputs15,inputs16,inputs17],
                  outputs=[output])
    return model
from keras import optimizers

### loss自定义
import keras.backend as K

def mseee(y_true, y_pred):
    y1=y_true[0];y2=y_true[1];y3=y_true[2];y4=y_true[3]
    y5 = y_true[4];y6=y_true[5];y7=y_true[6];y8=y_true[7]
    y9 = y_true[8];y10=y_true[9];y11=y_true[10];y12=y_true[11]
    y13 = y_true[12];y14=y_true[13];y15=y_true[14];y16=y_true[15]
    pred_y1 = y_pred[0];pred_y2 = y_pred[1];pred_y3 = y_pred[2];pred_y4 = y_pred[3]
    pred_y5 = y_pred[4];pred_y6 = y_pred[5];pred_y7 = y_pred[6];pred_y8 = y_pred[7]
    pred_y9 = y_pred[8];pred_y10 = y_pred[9];pred_y11 = y_pred[10];pred_y12 = y_pred[11]
    pred_y13 = y_pred[12];pred_y14 = y_pred[13];pred_y15 = y_pred[14];pred_y16 = y_pred[15]
    return K.mean((y1-pred_y1)**2+(y2-pred_y2)**2+(y3-pred_y3)**2+(y4-pred_y4)**2+(y5-pred_y5)**2+(y6-pred_y6)**2+(y7-pred_y7)**2+(y8-pred_y8)**2
                  +(y9-pred_y9)**2+(y10-pred_y10)**2+(y11-pred_y11)**2+(y12-pred_y12)**2+(y13-pred_y13)**2+(y14-pred_y14)**2+(y15-pred_y15+(y16-pred_y16)**2)**2)


adam = optimizers.Adam(lr=0.001, decay=1e-6)
model = attention_model()
model.compile(loss=mseee, optimizer='adam', metrics=['accuracy'])
model.summary()

from sklearn import preprocessing

data = pd.read_excel(r"D:\python_student\多输入和多输出\FCNDL\data\成渝城市群\PM25输入.xlsx",sheet_name="Sheet1")
data1 = data.values
X = data1[:, 1:-16]
Y = data1[:, -16:]

input_max = np.max(X)
input_min = np.min(X)
# data_x  = X
data_x = preprocessing.minmax_scale(X)

data2 = pd.read_excel(r"D:\python_student\多输入和多输出\FCNDL\data\成渝城市群\PM25输入.xlsx",sheet_name="Sheet2")
data2 =data2.values
data2 = data2[:,1:]

nn  = data2.shape[1]
city1 = []
city2 = []
city3 = []
city4 = []
city5 = []
city6 = []
city7 = []
city8 = []
city9 = []
city10 = []
city11 = []
city12 = []
city13 = []
city14 = []
city15 = []
city16 = []
for i in range(0,nn,16):
    a1 = data2[:,i];a2 = data2[:,i+1];a3 = data2[:,i+2];a4 = data2[:,i+3]
    a5 = data2[:,i+4]; a6 = data2[:,i + 5]; a7 = data2[:,i + 6]; a8 = data2[:,i + 7]
    a9 = data2[:,i+8]; a10 = data2[:,i + 9]; a11 = data2[:,i + 10]; a12 = data2[:,i + 11]
    a13 = data2[:,i+12]; a14 = data2[:,i + 13]; a15 = data2[:,i + 14]; a16 = data2[:,i + 15]
    city1.append(a1);city2.append(a2);city3.append(a3);city4.append(a4)
    city5.append(a5); city6.append(a6); city7.append(a7); city8.append(a8)
    city9.append(a9); city10.append(a10); city11.append(a11); city12.append(a12)
    city13.append(a13); city14.append(a14); city15.append(a15); city16.append(a16)

city1 = pd.DataFrame(city1);city2 = pd.DataFrame(city2);city3 = pd.DataFrame(city3);city4 = pd.DataFrame(city4)
city5 = pd.DataFrame(city5);city6 = pd.DataFrame(city6);city7 = pd.DataFrame(city7);city8 = pd.DataFrame(city8)
city9 = pd.DataFrame(city9);city10 = pd.DataFrame(city10);city11 = pd.DataFrame(city11);city12 = pd.DataFrame(city12)
city13 = pd.DataFrame(city13);city14 = pd.DataFrame(city14);city15 = pd.DataFrame(city15);city16 = pd.DataFrame(city16)

city1=city1.T;city2=city2.T;city3=city3.T;city4=city4.T
city5=city5.T;city6=city6.T;city7=city7.T;city8=city8.T
city9=city9.T;city10=city10.T;city11=city11.T;city12=city12.T
city13=city13.T;city14=city14.T;city15=city15.T;city16=city16.T



city1 = preprocessing.minmax_scale(city1)  # 标准化
city2 = preprocessing.minmax_scale(city2)  # 标准化
city3 = preprocessing.minmax_scale(city3)  # 标准化
city4 = preprocessing.minmax_scale(city4)  # 标准化
city5 = preprocessing.minmax_scale(city5)  # 标准化
city6 = preprocessing.minmax_scale(city6)  # 标准化
city7 = preprocessing.minmax_scale(city7)  # 标准化
city8 = preprocessing.minmax_scale(city8)  # 标准化
city9 = preprocessing.minmax_scale(city9)  # 标准化
city10 = preprocessing.minmax_scale(city10)  # 标准化
city11 = preprocessing.minmax_scale(city11)  # 标准化
city12 = preprocessing.minmax_scale(city12)  # 标准化
city13 = preprocessing.minmax_scale(city13)  # 标准化
city14 = preprocessing.minmax_scale(city14)  # 标准化
city15 = preprocessing.minmax_scale(city15)  # 标准化
city16 = preprocessing.minmax_scale(city16)  # 标准化




output_max = np.max(Y, 0)
output_min = np.min(Y, 0)
data_y = preprocessing.minmax_scale(Y)  # 标准化
# data_y = Y
# 数据集分割
data_len = len(data_x)
t = np.linspace(0, data_len, data_len)

train_data_ratio = 0.8  # Choose 80% of the data for training
train_data_len = int(data_len * train_data_ratio)

train_x = data_x[0:train_data_len]
train_y = data_y[0:train_data_len]
train_city1 = city1[0:train_data_len]
train_city2 = city2[0:train_data_len]
train_city3 = city3[0:train_data_len]
train_city4 = city4[0:train_data_len]
train_city5 = city5[0:train_data_len]
train_city6 = city6[0:train_data_len]
train_city7 = city7[0:train_data_len]
train_city8 = city8[0:train_data_len]
train_city9 = city9[0:train_data_len]
train_city10 = city10[0:train_data_len]
train_city11 = city11[0:train_data_len]
train_city12 = city12[0:train_data_len]
train_city13 = city13[0:train_data_len]
train_city14 = city14[0:train_data_len]
train_city15 = city15[0:train_data_len]
train_city16 = city16[0:train_data_len]
t_for_training = t[0:train_data_len]

test_x = data_x[train_data_len:]
test_y = data_y[train_data_len:]
test_city1 = city1[train_data_len:]
test_city2 = city2[train_data_len:]
test_city3 = city3[train_data_len:]
test_city4 = city4[train_data_len:]
test_city5 = city5[train_data_len:]
test_city6 = city6[train_data_len:]
test_city7 = city7[train_data_len:]
test_city8 = city8[train_data_len:]
test_city9 = city9[train_data_len:]
test_city10 = city10[train_data_len:]
test_city11 = city11[train_data_len:]
test_city12 = city12[train_data_len:]
test_city13 = city13[train_data_len:]
test_city14 = city14[train_data_len:]
test_city15 = city15[train_data_len:]
test_city16 = city16[train_data_len:]
t_for_testing = t[train_data_len - 1:]

INPUT_FEATURES_NUM = 16
OUTPUT_FEATURES_NUM = 16

city_FEATURES_NUM = 4
# 改变输入形状
train_x_tensor = train_x.reshape(-1, INPUT_FEATURES_NUM, 1)  # set batch size to 1

train_city1_tensor = train_city1.reshape(-1, 1, city_FEATURES_NUM)  # set batch size to 1
train_city2_tensor = train_city2.reshape(-1, 1, city_FEATURES_NUM) # set batch size to 1
train_city3_tensor = train_city3.reshape(-1, 1, city_FEATURES_NUM)  # set batch size to 1
train_city4_tensor = train_city4.reshape(-1, 1, city_FEATURES_NUM) # set batch size to 1
train_city5_tensor = train_city5.reshape(-1, 1, city_FEATURES_NUM)  # set batch size to 1
train_city6_tensor = train_city6.reshape(-1, 1, city_FEATURES_NUM) # set batch size to 1
train_city7_tensor = train_city7.reshape(-1, 1, city_FEATURES_NUM)  # set batch size to 1
train_city8_tensor = train_city8.reshape(-1, 1, city_FEATURES_NUM)  # set batch size to 1
train_city9_tensor = train_city9.reshape(-1, 1, city_FEATURES_NUM)  # set batch size to 1
train_city10_tensor = train_city10.reshape(-1, 1, city_FEATURES_NUM)  # set batch size to 1
train_city11_tensor = train_city11.reshape(-1, 1, city_FEATURES_NUM)  # set batch size to 1
train_city12_tensor = train_city12.reshape(-1, 1, city_FEATURES_NUM) # set batch size to 1
train_city13_tensor = train_city13.reshape(-1, 1, city_FEATURES_NUM) # set batch size to 1
train_city14_tensor = train_city14.reshape(-1, 1, city_FEATURES_NUM)  # set batch size to 1
train_city15_tensor = train_city15.reshape(-1, 1, city_FEATURES_NUM)  # set batch size to 1
train_city16_tensor = train_city16.reshape(-1, 1, city_FEATURES_NUM)  # set batch size to 1
train_y_tensor = train_y.reshape(-1, 1, OUTPUT_FEATURES_NUM)  # set batch size to 1



model11 = model.fit(x=[train_x_tensor,train_city1_tensor,train_city2_tensor,train_city3_tensor,train_city4_tensor,
                       train_city5_tensor,train_city6_tensor,train_city7_tensor,train_city8_tensor,train_city9_tensor,
                       train_city10_tensor,train_city11_tensor,train_city12_tensor,train_city13_tensor,train_city14_tensor,
                       train_city15_tensor,train_city16_tensor],
                    y=train_y_tensor, epochs=30)




test_city1_tensor = test_city1.reshape(-1, 1, city_FEATURES_NUM)  # set batch size to 1
test_city2_tensor = test_city2.reshape(-1, 1, city_FEATURES_NUM) # set batch size to 1
test_city3_tensor = test_city3.reshape(-1, 1, city_FEATURES_NUM)  # set batch size to 1
test_city4_tensor = test_city4.reshape(-1, 1, city_FEATURES_NUM) # set batch size to 1
test_city5_tensor = test_city5.reshape(-1, 1, city_FEATURES_NUM)  # set batch size to 1
test_city6_tensor = test_city6.reshape(-1, 1, city_FEATURES_NUM) # set batch size to 1
test_city7_tensor = test_city7.reshape(-1, 1, city_FEATURES_NUM)  # set batch size to 1
test_city8_tensor = test_city8.reshape(-1, 1, city_FEATURES_NUM)  # set batch size to 1
test_city9_tensor = test_city9.reshape(-1, 1, city_FEATURES_NUM)  # set batch size to 1
test_city10_tensor = test_city10.reshape(-1, 1, city_FEATURES_NUM)  # set batch size to 1
test_city11_tensor = test_city11.reshape(-1, 1, city_FEATURES_NUM)  # set batch size to 1
test_city12_tensor = test_city12.reshape(-1, 1, city_FEATURES_NUM) # set batch size to 1
test_city13_tensor =test_city13.reshape(-1, 1, city_FEATURES_NUM) # set batch size to 1
test_city14_tensor = test_city14.reshape(-1, 1, city_FEATURES_NUM)  # set batch size to 1
test_city15_tensor = test_city15.reshape(-1, 1, city_FEATURES_NUM)  # set batch size to 1
test_city16_tensor = test_city16.reshape(-1, 1, city_FEATURES_NUM)  # set batch size to 1





pred_y_for_test = model.predict([test_x,test_city1_tensor,test_city2_tensor,test_city3_tensor,test_city4_tensor,
                       test_city5_tensor,test_city6_tensor,test_city7_tensor,test_city8_tensor,test_city9_tensor,
                       test_city10_tensor,test_city11_tensor,test_city12_tensor,test_city13_tensor,test_city14_tensor,
                       test_city15_tensor,test_city16_tensor])



train_y_for_test = model.predict([train_x,train_city1_tensor,train_city2_tensor,train_city3_tensor,train_city4_tensor,
                       train_city5_tensor,train_city6_tensor,train_city7_tensor,train_city8_tensor,train_city9_tensor,
                       train_city10_tensor,train_city11_tensor,train_city12_tensor,train_city13_tensor,train_city14_tensor,
                       train_city15_tensor,train_city16_tensor])
#


train_y_for_test = np.squeeze(train_y_for_test)
pred_y_for_test = np.squeeze(pred_y_for_test)

for i in range(16):
    pred_y_for_test[:,i] = (output_max[i] - output_min[i]) * np.array(pred_y_for_test[:,i]) + output_min[i]


# pred_y_for_test = pred_y_for_test.reshape(np.size(pred_y_for_test),)

for i in range(16):
    train_y_for_test[:,i] = (output_max[i] - output_min[i]) * np.array(train_y_for_test[:,i]) + output_min[i]


for i in range(16):
    train_y[:, i] = (output_max[i] - output_min[i]) * np.array(train_y[:, i]) + output_min[i]


for i in range(16):
    test_y[:, i] = (output_max[i] - output_min[i]) * np.array(test_y[:, i]) + output_min[i]


##  训练集  ###
train_y1 = pd.DataFrame(train_y)
train_y1.to_csv("train_y1.csv")
train_y_for_test = pd.DataFrame(train_y_for_test)
train_y_for_test.to_csv("train_y_for_test.csv")


##  测试集  ###
test_y = pd.DataFrame(test_y)
test_y.to_csv('test_y2.csv')  # 数据存入csv,存储位置及文件名称
pred_y_for_test_data = pd.DataFrame(pred_y_for_test)  # 将验证集数据的预测值放进表格
pred_y_for_test_data.to_csv('pred_y_for_test2.csv')  # 数据存入csv,存储位置及文件名称


