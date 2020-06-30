#该篇是以char作为最小单元进行的lstm训练
import numpy
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

#构造训练数据集
seq_length = 2
#前置字母
x = []
#后置字母
y = []

n_vocab = 60
# 读入文本
raw_text = open('./data/Winston_Churchil.txt', encoding='utf-8').read()
raw_text = raw_text.lower()

# 字母一共有26个，我们采用One-Hot来编码出所有的字母，应该还有其他的特殊字符和数字
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

print(chars)
print(char_to_int)
print(int_to_char)
print(len(chars))
print(len(raw_text))

# for char in raw_text[0:100]:
#     print(char)
#     print(char_to_int[char])
#     x.append([char_to_int[char]])

for i in range(0, len(raw_text) - seq_length):
    given = raw_text[i:i + seq_length]
    predict = raw_text[i + seq_length]
    x.append([char_to_int[char] for char in given])
    y.append(char_to_int[predict])

# 打印一下训练的数据集
print(x[:3])
print(y[:3])

# 第一我们已经有了一个input的数字表达（index），我们要把它变成LSTM需要的数组格式： [样本数，时间步伐，特征]
# 第二，对于output，我们在Word2Vec里学过，用one-hot做output的预测可以给我们更好的效果，相对于直接预测一个准确的y数值的话。

n_patterns = len(x)
n_vocab = len(chars)

# 把x变成LSTM需要的样子
x = numpy.reshape(x, (n_patterns, seq_length, 1))
print(x[11])
# 简单的normal到0-1之间
x = x / float(n_vocab)
# 把output变成One-Hot
y = np_utils.to_categorical(y)
print(x[11])
print(y[11])

def  train():
    #构造模型 LSTM
    model =  Sequential()
    model.add(LSTM(1024,input_shape=(x.shape[1],x.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1],activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    model.fit(x, y, nb_epoch=500, batch_size=4096)

    # 模型保存
    model.save(filepath="./models/Lstm_Char_rnn.h5")

# 验证数据集
def predict_next(input_array,model):
    x = numpy.reshape(input_array,(1,seq_length,1))
    x = x/float(n_vocab)
    y = model.predict(x)
    return y

#字符转换成编码
def string_to_index(raw_input):
    res = []
    start = len(raw_input)-seq_length if len(raw_input)>=seq_length else 0
    for c  in raw_input[start:]:
        res.append(char_to_int[c])
    if (seq_length-len(raw_input))>0:
       res = numpy.pad(res,(seq_length-len(raw_input),0),'constant', constant_values=(0,0))
    print(res)
    return res
#编码转换成字符
def y_to_char(y):
    latgest_index  = y.argmax()
    c = int_to_char[latgest_index]
    return c

#合并测试
def generate_article(init,rounds=2):
    model = load_model("./models/Lstm_Char_rnn_5.h5")
    in_string = init.lower()
    for i in range(rounds):
        n = y_to_char(predict_next(string_to_index(in_string),model))
        in_string +=n
    return in_string




if __name__=="__main__":
    # train()
    # 开始测试
    init = 'winsto'
    print(len(init))
    article = generate_article(init)
    print(article)

    # # res =[3,4,5,6]
    # # res = numpy.pad(res, (1, 0), 'constant', constant_values=(0, 0))
    # # print(res)

