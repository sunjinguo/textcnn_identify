#! /usr/bin/env python
#coding=utf-8
from cnn_detector_1 import *

def cnn_predict():
    text='2018年10月9号，这是一条测试语句！'
    client=cnn_detector()
    res=client.ad_detect(text)
    print text
    print res
    return text,res

def get_file_predicts():
    path='../data/train_data_1.txt'
    client=cnn_detector()
    with open(path)as rf,open('./datas.txt','w')as wf:
        for line in rf:
            msg=line.strip().decode('utf-8').split('__label__')
            if len(msg)<2:continue
            res=client.ad_detect(msg[0])
            if int(res)==int(msg[1]):continue
            wf.write('\t'.join([msg[0].encode('utf-8'),str(msg[1]),str(res)])+'\n')


if __name__=='__main__':
    get_file_predicts()

