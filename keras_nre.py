from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from relation import load_data

(X_train, y_train), (X_test, y_test) = load_data()

X_train = X_train.reshape(-1, 1, 60, 312)  # (10000, 60, 312) -> (10000,1,60,312)
X_test = X_test.reshape(-1, 1, 60, 312)  # (2000, 28, 28) -> (2000,1,60,312)
# y_train = np_utils.to_categorical(y_train, 10)  # (60000,) -> (60000, 10)
# y_test = np_utils.to_categorical(y_test, 10)  # (10000,) -> (10000, 10)

model = Sequential()
model.add(
    Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=(1, 60, 312)))  # 卷积
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))  # 池化
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'))  # 卷积
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))  # 池化
model.add(Flatten())  # 打平
model.add(Dense(512))  # 全连接
model.add(Activation('relu'))  # relu激活函数
model.add(Dense(35))  # 全连接(类别数)
model.add(Activation('softmax'))  # softmax激活函数

adam = Adam(lr=1e-4)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=2, batch_size=64)  # 训练模型

loss, accuracy = model.evaluate(X_test, y_test)  # 评估模型
print('loss: ', loss, 'accuracy: ', accuracy)
