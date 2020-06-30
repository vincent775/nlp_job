from keras.layers import Input,Dense
from keras.models import Model
from sklearn.cluster import KMeans
import numpy as np


class ASCIIAutoencoder():
    """基于字符串的Autoencoder"""
    def __init__(self,sen_len=512,encoding_dim=32,epoch=50,val_tatio=0.3):
        """
        Init.
        :param sen_len:  把sentences pad成相同的长度
        :param encoding_dim: 压缩后的维度dim
        :param epoch: 要跑多少epoch
        :param kmeanmodel: 简单的KNN clustering模型
        """
        self.sen_len = sen_len
        self.encodeing_dim = encoding_dim
        self.autoencoder  = None
        self.encoder =  None
        self.kmeanmodel = KMeans(n_clusters=2)
        self.epoch = epoch
    def fit(self,x):
        """
        模型构建
        :param x: input text
        :return:
        """
        x_train = self.preprocess(x,length=self.sen_len)
        input_text = Input(shape=(self.sen_len))

        # "encoded" 没经过一层，都被刷新成小一点的“压缩后表达式”
        encoded = Dense(1024,activation='tanh')(input_text)
        encoded = Dense(512, activation='tanh')(encoded)
        encoded = Dense(128, activation='tanh')(encoded)
        encoded = Dense(self.encodeing_dim, activation='tanh')(encoded)

        # “decoded” 就是把刚刚压缩完的东西，给反过来还原成input_text
        decoded = Dense(128,activation='tanh')(encoded)
        decoded = Dense(512, activation='tanh')(decoded)
        decoded = Dense(1024, activation='tanh')(decoded)
        decoded = Dense(self.sen_len, activation='sigmoid')(decoded)

        # 整个从大到小再到大的model，叫做autoencoder
        self.autoencoder =  Model(input=input_text,output=decoded)

        # 那么只从大到小 （也就是一般的model）就叫encoder
        self.encoder = Model(input=input_text,output=encoded)

        #同理  我们搞一个decoder出来，也就是从小到大的model
        #首先encoded的input_size 预留号
        encoded_input = Input(shape=(1024,))
        #autoencoder的最后一层，就应该是decoder的第一层
        decoder_layer = self.autoencoder.layers[-1]
        #然后我们从头到尾链接起来，就是一个decoder了
        decoder = Model(input=encoded_input,output=decoder_layer(encoded_input))

        #compile
        self.autoencoder.compile(optimizer='adam',loss='mse')

        #跑起来
        self.autoencoder.fit(x_train,x_train,nb_epoch=self.epoch,batch_size=1000,shuffle=True,)

        #这一部分是自己拿自己train一下knn，一件简单的基于距离的分类器
        x_train = self.encoder.predict(x_train)
        self.kmeanmodel.fit(x_train)

    def predict(self,x):
        """
        做预测
        :param x: input text
        :return: predicttions
        """
        # 同理，第一步 把输入的参数搞成ASCII化，并且长度相同
        x_test = self.encoder.predict(x,length=self.sen_len)
        # 然后用encoder 把test集合给压缩
        x_test = self.encoder.predict(x_test)
        # KNN  给分类出来
        preds = self.kmeanmodel.predict(x_test)

        return preds
    def preprocess(self, s_list, length=256):
        return s_list





if __name__ == '__main__':
    x= "teacher"
    X_train = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    tyt = ASCIIAutoencoder()
    tyt.fit(X_train)
    ASCIIAutoencoder.predict(X_train)
