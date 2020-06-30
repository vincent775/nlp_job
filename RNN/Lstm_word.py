#该篇是以word作为最小单元进行的lstm训练
import os
import numpy as np
import nltk
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from gensim.models.word2vec import Word2Vec
from keras.models import load_model

#加载数据
raw_text = ""
for file in os.listdir('./data/'):
    if file.endswith(".txt"):
        raw_text += open("./data/"+file,errors='ignore',encoding='utf-8').read()+'\n\n'
# raw_text = open("./data/Winston_Churchil.txt",encoding='utf-8').read()
raw_text =  raw_text.lower()
sentensor = nltk.data.load('./tokenizers/punkt/english.pickle')
sents = sentensor.tokenize(raw_text)
corpus = []
for sen in sents:
    corpus.append(nltk.word_tokenize(sen))
print('总的句子数'+str(len(corpus)))
print(corpus[:3])

#w2v 乱炖
w2v_model = Word2Vec(corpus,size=128,window=5,min_count=5,workers=8)

print(w2v_model['office'])

#接下来还是以单字的训练方式进行
raw_input = [item for sublist in corpus for item in sublist]
print('总单词数'+str(len(raw_input)))
print(raw_input[12])

text_stream = []
vocab = w2v_model.wv.vocab
for word in raw_input:
    if word in vocab:
        text_stream.append(word)
print('...'+str(len(text_stream)))

#构造字母和数字下标
seq_length = 10
x = []
y = []
for i in range(0, len(text_stream) - seq_length):
    # print(i)
    given = text_stream[i:i + seq_length]
    predict = text_stream[i + seq_length]
    x.append(np.array([w2v_model[word] for word in given]))
    y.append(w2v_model[predict])
print(x[10])
print(y[10])
print(len(x))
print(len(y))
print(len(x[12]))
print(len(x[12][0]))
print(len(y[12]))
print(type(x))
print(type(y))
x = np.reshape(x,(-1,seq_length,128))
y = np.reshape(y,(-1,128))

#接下来做两件事，
#1。把w2c变成LSTM需要的数组格式：【样本数，时间步伐，特征】
#2.对于output,我们直接用128维输出
def train():
    #构建LSTM模型
    model = Sequential()
    model.add(LSTM(1024,dropout_W=0.2,dropout_U=0.2,input_shape=(seq_length,128)))
    model.add(Dropout(0.2))
    model.add(Dense(128,activation='sigmoid'))
    model.compile(loss='mse',optimizer='adam')
    model.fit(x,y,nb_epoch=50,batch_size=4096)
    # 模型保存
    model.save(filepath="./models/Lstm_Word_rnn.h5")

def predict_next(input_array,model):
    x = np.reshape(input_array, (-1,seq_length,128))
    y = model.predict(x)
    return y

def string_to_index(raw_input):
    raw_input = raw_input.lower()
    input_stream = nltk.word_tokenize(raw_input)
    res = []
    for word in input_stream[(len(input_stream)-seq_length):]:
        # print(w2v_model['office'])
        print(word)
        if word in vocab:
            res.append(w2v_model[word])
    if (seq_length - len(res)) > 0:
        # res = np.pad(res, (seq_length - len(res), 0), 'constant', constant_values=(0, 0))
        res = np.pad(res, ((seq_length - len(res),0),(0,0)), 'constant', constant_values=(0, 0))
    print('Res的长度：'+str(len(res)))
    print(res)
    return res

def y_to_word(y):
    word = w2v_model.most_similar(positive=y, topn=1)
    return word
def generate_article(init, rounds=2):
    model = load_model("./models/Lstm_Word_rnn.h5")
    in_string = init.lower()
    for i in range(rounds):
        n = y_to_word(predict_next(string_to_index(in_string),model))
        in_string += ' ' + n[0][0]
    return in_string

if __name__=='__main__':
    train()
    # 开始测试
    # init = 'lack of medical care. The only instruments they possessed '
    # print(len(init))
    # article = generate_article(init)
    # print(article)