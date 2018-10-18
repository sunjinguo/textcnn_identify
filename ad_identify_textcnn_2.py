#! /usr/bin/env python
#coding=utf-8
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from ad_util import *
import data_helpers
from tensorflow.contrib import learn
import csv
from bs4 import BeautifulSoup as BS
import re
import jieba

#constants
re_NUM = re.compile(ur'([-]?[\d]+[\d\%.,]*[ ]?[百千万亿]{,3})')
re_COMBINE = re.compile(ur'(?![0-9]+[^0-9A-Za-z\-]+)(?![a-zA-Z]+[^0-9A-Za-z\-]+)[0-9A-Za-z\-]{2,}')
re_DATE=re.compile(ur'\d{2,4}年|\d{2,4}年\d{1,2}月\d{1,2}日|\d{2,4}年\d{1,2}月\d{1,2}号|\d{2,4}年\d{1,2}月|\d{1,2}月\d{1,2}日|\d{1,2}月\d{1,2}号|\d{2,4}\.\d{1,2}\.\d{1,2}|\d{2,4}\-\d{1,2}\-\d{1,2}|\d{1,2}日|\d{1,2}号|\d{1,2}月|\d{1,2}:\d{1,2}')
re_TIME=re.compile(ur'^(([0-1]?[0-9])|([2][0-3])):([0-5]?[0-9])(:([0-5]?[0-9]))?$')
re_HTTP = re.compile(ur'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
re_PUNC=re.compile(u'[\t\n .;,:：、?！？/%#*&$\'"，。()（）【】\[\]|\-{}<>《》‘’“”「」『』▲★△;；]')



def get_sens(text):
    '''将文本切分为段落'''
    text=text.strip()
    if not isinstance(text,unicode):text=text.decode('utf-8')
    sentences=[]
    if text.startwith('<'):
        try:
            soup=BS(text)
        except:
            return []
        p_list=soup.find_all('p')
        for p in p_list:
            sen=p.get_text()
            if not sen.strip():continue
            sentences.append(sen.replace(' ','').lower())
    else:
        sens=text.split('\n')
        for sen in sens:
            if not sen.strip():continue
            sentences.append(sen.replace(' ','').lower())
    return sentences

def text_preprocess(text):
    if not isinstance(text,unicode):text=text.decode('utf-8')
    content=re.sub(re_HTTP,'',text)
    content=re.sub(re_PUNC,'',content)
    content=re.sub(re_DATE,'TAGDATE',content)
    content=re.sub(re_TIME,'TAGTIME',content)
    content=re.sub(re_COMBINE,'TAGCOMBINE',content)
    content=re.sub(re_NUM,'TAGNUMBER',content)
    return content

def get_tokenStr(text):
    #jieba.load_userdict('./jieba_mydict.txt')
    word_list=jieba.cut(text)
    textStr=' '.join(word_list)
    return textStr

def load_textcnn_model():
    
    return


# Parameters
tf.flags.DEFINE_string("checkpoint_dir", "../runs/train_model/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


# vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
print 'vocab:',vocab_path
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
checkpoint_file=os.path.join(FLAGS.checkpoint_dir,'model-47100')

def main():
    graph = tf.Graph()
    with graph.as_default():
        session_conf=tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement)
        sess=tf.Session(config=session_conf)
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            scores=graph.get_operation_by_name('output/scores').outputs[0]
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            text='1982年综合网络60如有侵权联系删除####****'
            text=text_preprocess(text)
            text=get_tokenStr(text)
            print text,type(text)
            x=get_vector(text,vocab_processor)
            score=sess.run(scores,feed_dict={input_x:[x],dropout_keep_prob:1.0})[0]
            res=sess.run(predictions,feed_dict={input_x:[x],dropout_keep_prob:1.0})[0]
            print text.replace(' ',''),score,res


if __name__=='__main__':
    main()
