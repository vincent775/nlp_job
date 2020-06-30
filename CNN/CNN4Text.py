# 有两种⽅方法可以做CNN for text
#  每种⽅方法⾥里里⾯面，也有各种不不同的玩法思路路
#  效果其实基本都差不不多，
#  我这⾥里里讲2种最普遍的：
#  1. ⼀一维的vector [...] + ⼀一维的filter [...]
#  这种⽅方法有⼀一定的信息损失（因为你average了了向量量），但是速度快，效果也没太差。
#  2. 通过w2v或其他⽅方法，做成⼆二维matrix，当做图⽚片来做。
#  这是⽐比较“讲道理理”的解决⽅方案，但是就是慢。。。放AWS上太烧钱了了。。。
#  1. 1D CNN for Text
#  ⼀一个IMDB电影评论的例例⼦子

import numpy as np
np.random.seed(1337)

from keras.preprocessing import sequence
from keras.preprocessing import text
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Embedding
from keras.layers import Convolution1D,MaxPool1D
from keras.datasets import imdb
import matplotlib.pyplot as plt

# set parameters:
max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
nb_filter = 250
filter_length = 3
hidden_dims = 250
nb_epoch = 2


def load_data():
    # 自带的数据集，load IMDB  data
    (X_train, Y_train), (X_test, Y_test) = imdb.load_data(path='E:\\project\\nlp_job\\CNN\\data\\imdb.npz',
                                                          nb_words=max_features)
    # 这个数据集是已经搞好的BOW，长这样
    # [123,2,0,45,32,1212,344,4,...]
    # 简单的把它们搞成相同的长度，不够的补0.太多的砍掉

    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    # 这里可以换成word2vec的vecor 它们就是天然的相同长度了
    return X_train, Y_train,X_test, Y_test

def train(X_train, Y_train,X_test, Y_test):


    #初始化我们的sequential model（指的是线性排列的layer）
    model = Sequential()

    #亮点来了，这里你需要一个Embedding layer来把你的input word indexes
    #变成tensor vectors：比如 如果dim是3 那么：
    #[[2],[123],...] -->[[0.1,0.4,0.21],[0.2,0.4,0.13],...]
    model.add(Embedding(max_features,embedding_dims,input_length = maxlen,dropout=0.2))
    # 这一步对于直接用BOW（比如这个IMDB的数据集）很方便，膳食对我们自己的word vector，就不太友好了，可以选择跳过他

    #现在可以添加一个Conv layer了
    model.add(Convolution1D(nb_filter=nb_filter,filter_length=filter_length,border_mode='valid',activation='relu',subsample_length=1))

    #后面跟一个maxpooling
    model.add(MaxPool1D(pool_length=model.output_shape[1]))
    #pool出来的结果，就二十类似于一堆的小VEC
    # 把他们粗暴的flatten一下 （就是横着连成一起）
    model.add(Flatten())

    #接下来就是简单的MLP了
    # 在keras里面，普通的layer用dense表示
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    #最终层
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # 如果面对的是时间序列的话，这里也是可以把layers 换成LSTM

    #compile
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    #跑起来
    history = model.fit(X_train,Y_train,batch_size=batch_size,nb_epoch=nb_epoch,validation_data=(X_test,Y_test))
    #这里有个fault啊，不可以拿testset做validation的，这只是一个简单的事例，为了跑起来方便

    #模型保存
    # 模型保存
    model.save(filepath="./models/demo_imdb_rnn.h5")

    #测试
    result = model.predict(X_test)
    print(result)
    show_train_history(history)


# 绘制训练 & 验证的准确率值
def show_train_history(train_history):
    print(train_history.history.keys())
    print(train_history.epoch)
    plt.plot(train_history.history['acc'])
    plt.plot(train_history.history['val_acc'])
    plt.title("model accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()

    plt.plot(train_history.history['loss'])
    plt.plot(train_history.history['val_loss'])
    plt.title("model loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

#使用模型进行预测
def predict(x):
    # x_text
    # x = gen_predict_data(path)
    model = load_model("./models/demo_imdb_rnn.h5")
    y = model.predict(x)
    print(y)
    y = model.predict_classes(x)
    print(y)
    print(RESULT[y[0][0]])
# def predict(path):
#     x = gen_predict_data(path)
#     model = load_model("./models/demo_imdb_rnn.h5")
#     y = model.predict(x)
#     print(y)
#     y = model.predict_classes(x)
#     print(y)
#     print(RESULT[y[0][0]])
# def gen_predict_data(path):
#     sent = prepross(path)
#     x_train, t_train = imdb_load("train")
#     token = text.Tokenizer(num_words=max_features)
#     token.fit_on_texts(x_train)
#     x = token.texts_to_sequences([sent])
#     x = sequence.pad_sequences(x, maxlen=maxlen)
#     return x

RESULT = {1: 'pos', 0: 'neg'}


if __name__ =="__main__":
    #获取数据
    X_train, Y_train, X_test, Y_test =  load_data()
    #训练模型
    # train(X_train, Y_train, X_test, Y_test)
    X_train=[[2,3,4,5,6,1,2,4,5,2,34,56,34,12],[12,13,13,234,12,34,45,67,15,45,67,78,89,76]]
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    ss = X_test[0:2]
    predict(ss)

