#coding=utf-8
import numpy as np
from gensim.models import KeyedVectors
import tensorflow as tf

def load_data(path):
    texts=[]
    with open(path,'r')as f:
        for line in f:
            text=line.decode('utf-8').strip().split('__label__')[0].strip()
            texts.append(text)
            #yield text.decode('utf-8')
    return texts

def load_datas(path):
    with open(path,'r')as f:
        for line in f:
            text=line.strip().split('__label__')[0].strip()
            yield text.decode('utf-8')

def load_labels(path):
    labels=[]
    with open(path)as f:
        for line in f:
            l=int(line.strip().split('__label__')[-1])
            if l==1:
                label=np.array([0,1])
            else:
                label=np.array([1,0])
            labels.append(label)
    return np.array(labels)


def get_vectors(vocab_processor,path):
    texts=load_datas(path)
    vectors=[]
    for text in texts:
        vector=list(vocab_processor.transform([text]))[0]
        vectors.append(vector)
    return np.array(vectors)

def get_vector(text,vocab_processor):
    vector=list(vocab_processor.transform([text]))[0]
    return vector

def load_word2vec():
    path='/data0/nfs_data/nlp/word2vec/word2vec_128.bin'
    return KeyedVectors.load_word2vec_format(path,binary=True)

def get_embeddings(vocab_processor,model,embedding_size=128):
    embeddings=[]
    vocab_size=len(vocab_processor.vocabulary_)
    for i in xrange(vocab_size):
        word=vocab_processor.vocabulary_.reverse(i)
        try:
            vec=model[word]
        except:
            vec=np.random.rand(embedding_size)
        embeddings.append(np.float32(vec))
    #print(np.array(embeddings).shape)
    return np.array(embeddings)

