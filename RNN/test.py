from gensim.models.word2vec import Word2Vec
import numpy as  np
wr = [['岛津','百度'],['岛津','百度'],['岛津','百度'],['天津','上海']]



# w2v_model = Word2Vec(corpus,size=128,window=5,min_count=5,workers=4)
w2C = Word2Vec(wr, min_count=1)
vocab = w2C.wv.vocab
for word in wr:
    if word in vocab:
        print(word)
print(w2C)