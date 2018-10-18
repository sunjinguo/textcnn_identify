#! /usr/bin/env python
#coding=utf-8
import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import jieba
import re
from constants import *

class cnn_detector(object):
    '''textcnn模型识别广告'''
    def __init__(self):
        '''
            初始化session,加载模型及相关参数
        '''
        # 加载词表
        self.vocab_processor=learn.preprocessing.VocabularyProcessor.restore(CNN_VOCAB_PATH)
        
        #构建graph和session,加载模型
        self.graph=tf.Graph()
        with self.graph.as_default():
            session_conf=tf.ConfigProto(
                    allow_soft_placement=True,
                    log_device_placement=False)
            self.session=tf.Session(config=session_conf)
            saver=tf.train.import_meta_graph(CNN_META_PATH)
            saver.restore(self.session,CNN_MODEL_PATH)
    
    def get_vector(self,text):
        '''
            将段落转化为词典id序列
        '''
        tokens=self.get_tokenstr(text)
        return list(self.vocab_processor.transform([tokens]))
    
    def text_preprocess(self,text):
        '''
            文本预处理，短链、符号过滤及数字、日期，字母组合替换
        '''
        if not isinstance(text,unicode):text=text.decode('utf-8')
        content=re.sub(RE_HTTP,'',text)
        content=re.sub(RE_TIME,'TAGDATE',content)
        content=re.sub(RE_DATE,'TAGDATE',content)
        content=re.sub(RE_COMBINE,'TAGCOMBINE',content)
        content=re.sub(RE_NUM,'TAGNUMBER',content)
        content=re.sub(RE_PUNC,'',content)
        return content

    def get_tokenstr(self,text):
        '''
            获取分词后段落
        '''
        jieba.load_userdict(JIEBA_USERDICT_PATH)
        word_list=jieba.cut(text)
        return ' '.join(word_list)
    
    def ad_detect(self,text):
        '''
            判断文本是否为广告,返回类别score
        '''
        content=self.text_preprocess(text)
        x=self.get_vector(content)
        input_x=self.graph.get_operation_by_name('input_x').outputs[0]
        dropout_keep_prob=self.graph.get_operation_by_name('dropout_keep_prob').outputs[0]
        predictions=self.graph.get_operation_by_name('output/predictions').outputs[0]
        scores=self.graph.get_operation_by_name('output/scores').outputs[0]
        prediction,score=self.session.run([predictions,scores],feed_dict={input_x:x,dropout_keep_prob:1.0})
        print 'prediction:', prediction[0]
        print 'score:',score[0]
        return prediction[0],score[0]
    
