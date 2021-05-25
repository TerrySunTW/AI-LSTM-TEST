import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector,Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

start_time= time.time()

stockdf = pd.read_csv('LSTM_dataset.csv',index_col=0)
#remove any empty column data
stockdf.dropna(how='any', inplace=True)

from sklearn import preprocessing
from keras.utils import np_utils
#data normalization object, please google MinMaxScaler if need more information
min_max_scaler = preprocessing.MinMaxScaler()

newdf=stockdf.copy()
flagdf=stockdf.copy()
#data normalization to modify data to 0~1 base on max and min value.
newdf['open'] = min_max_scaler.fit_transform(stockdf.open.values.reshape(-1,1))
newdf['low'] = min_max_scaler.fit_transform(stockdf.low.values.reshape(-1,1))
newdf['high'] = min_max_scaler.fit_transform(stockdf.high.values.reshape(-1,1))
newdf['close'] = min_max_scaler.fit_transform(stockdf.close.values.reshape(-1,1))
newdf['volume'] = min_max_scaler.fit_transform(stockdf.volume.values.reshape(-1,1))

#print(newdf)

import numpy as np
datavalue = newdf.values #only get value?
result = []
#print(newdf)
#print(datavalue)

time_frame = 10
for index in range(len(datavalue)-(time_frame)):
    result.append(datavalue[index: index+(time_frame)])

new_result = np.array(result) #why need another np.array to change value to np array
#print(result)
#print(new_result)
#print(new_result.shape)
#print(new_result)

#use 90% datas for training
number_train = round(0.9 * new_result.shape[0])

#get train data from new result(why only 0~5 columns?)
X_train = new_result[:int(number_train),:-1,0:5]

#get label from new result
Y_train = new_result[:int(number_train),-1][:,-1]

#always use onehot result to train AI model
Y_train_onehot = np_utils.to_categorical(Y_train)


X_test =  new_result[int(number_train):,:-1,0:5]
Y_test = new_result[int(number_train):,-1][:,-1]
Y_test_onehot = np_utils.to_categorical(Y_test)

print(X_train.shape)

model = Sequential()
model.add(LSTM(256, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(LSTM(256, return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.add(Flatten())
model.add(Dense(5,activation='linear'))
model.add(Dense(1,activation='linear'))
model.compile(loss="mean_absolute_error", optimizer="adam",metrics=['mean_absolute_error'])
model.summary()


history = model.fit(X_train, Y_train, epochs=10, batch_size=5, validation_split=0.1,shuffle=True,verbose=2)

end_time=time.time()

print("execution time:%F sec" % (end_time - start_time))
