#coding=utf-8
import os
import re

#textcnn 模型相关
CNN_DIR_PATH='/data0/home/jinguo3/workspace/jinguo3/TextCNN/cnn-text-classification-tf-master/AD_Identify/models/cnn'
CNN_VOCAB_PATH=os.path.join(CNN_DIR_PATH,'vocabs/vocab')
CNN_META_PATH=os.path.join(CNN_DIR_PATH,'model-47100.meta')
CNN_MODEL_PATH=os.path.join(CNN_DIR_PATH,'model-47100')
JIEBA_USERDICT_PATH='/data0/home/jinguo3/workspace/jinguo3/TextCNN/cnn-text-classification-tf-master/AD_Identify/config/jieba_mydict.txt'

#文本处理正则表达式
RE_NUM = re.compile(ur'([-]?[\d]+[\d\%.,]*[ ]?[百千万亿]{,3})')
RE_COMBINE = re.compile(ur'(?![0-9]+[^0-9A-Za-z\-]+)(?![a-zA-Z]+[^0-9A-Za-z\-]+)[0-9A-Za-z\-]{2,}')
RE_DATE=re.compile(ur'\d{2,4}年|\d{2,4}年\d{1,2}月\d{1,2}日|\d{2,4}年\d{1,2}月\d{1,2}号|\d{2,4}年\d{1,2}月|\d{1,2}月\d{1,2}日|\d{1,2}月\d{1,2}号|\d{2,4}\.\d{1,2}\.\d{1,2}|\d{2,4}\-\d{1,2}\-\d{1,2}|\d{1,2}日|\d{1,2}号|\d{1,2}月|\d{1,2}:\d{1,2}')
RE_TIME=re.compile(ur'^(([0-1]?[0-9])|([2][0-3])):([0-5]?[0-9])(:([0-5]?[0-9]))?$')
RE_HTTP = re.compile(ur'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
RE_PUNC=re.compile(u'[\t\n .;,:：、?！？/%#*&$\'"，。()（）【】\[\]|\-{}<>《》‘’“”「」『』▲★△;；]')
