#!C:\Users\60067527\Anaconda3\envs\py36
#-*- coding: utf-8 -*-
from __future__ import absolute_import

import sys, os, io, logging

import numpy as np
import tensorflow as tf

from PIL import Image
from six import BytesIO as IO

from .bucketdata import BucketData

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH,
                                  #'../labels/bank_labelsSW.txt')
                                  #'../labels/bank_labelsS.txt')								  
                                  '../labels/bank_labels.txt')	

try:
    TFRecordDataset = tf.data.TFRecordDataset  # pylint: disable=invalid-name
except AttributeError:
    TFRecordDataset = tf.contrib.data.TFRecordDataset  # pylint: disable=invalid-name



class DataGen(object):
    GO_ID = 1
    EOS_ID = 2
    IMAGE_HEIGHT = 32



    label_file = DEFAULT_LABEL_FILE
    with io.open(label_file, 'r', encoding='utf-8') as f:
        labels = f.read().splitlines()
        
    CHARMAP =labels


	
    #CHARMAP = ['', '', ''] + list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    #CHARMAP = ['', '', ''] + list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ신한국민하나우리기업농협.:,-()')
    #CHARMAP = ['', '', ''] + list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ신한') 
	# 신 49888,  한 54620
	
    @staticmethod
    def set_full_ascii_charmap():
        #DataGen.CHARMAP = ['', '', ''] + [i for i in range(32, 127)]
        DataGen.CHARMAP = ['', '', ''] + [chr(i) for i in range(32, 127)]

    def __init__(self,
                 annotation_fn,
                 buckets,
                 epochs=1000,
                 max_width=None):
        """
        :param annotation_fn:
        :param lexicon_fn:
        :param valid_target_len:
        :param img_width_range: only needed for training set
        :param word_len:
        :param epochs:
        :return:
        """
        self.epochs = epochs
        self.max_width = max_width

        self.bucket_specs = buckets
        self.bucket_data = BucketData()

        dataset = TFRecordDataset([annotation_fn])
        dataset = dataset.map(self._parse_record)
        dataset = dataset.shuffle(buffer_size=10000)
        self.dataset = dataset.repeat(self.epochs)

    def clear(self):
        self.bucket_data = BucketData()

    def gen(self, batch_size):
        #logging.info(' data_gen.gen()') 
        dataset = self.dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
 		
        images, labels, comments = iterator.get_next()
  	
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
   
            while True:
                try:				
                    raw_images, raw_labels, raw_comments = sess.run([images, labels, comments])
  
                    for img, lex, comment in zip(raw_images, raw_labels, raw_comments):
                        
 
                        if self.max_width and (Image.open(IO(img)).size[0] <= self.max_width):
                             						
                            word = self.convert_lex(lex)
                             
                            bucket_size = self.bucket_data.append(img, word, lex, comment)
  	
                            if bucket_size >= batch_size:
                                bucket = self.bucket_data.flush_out(
                                    self.bucket_specs,
                                    go_shift=1)
                                yield bucket

                except tf.errors.OutOfRangeError:
                    break
 
        self.clear()

    def convert_lex(self, lex):

        if sys.version_info >= (3,):
            lex = lex.decode('utf-8')
            #lex = lex.decode('iso-8859-1')
        
        #assert len(lex) < self.bucket_specs[-1][1]
		#return np.array(
        #    [self.GO_ID] + [self.CHARMAP.index(char) for char in lex] + [self.EOS_ID],
        #    dtype=np.int32)
         
        GO_ID = 1
        EOS_ID = 2



        label_file = DEFAULT_LABEL_FILE
        with io.open(label_file, 'r', encoding='utf-8') as f:
            labels = f.read().splitlines()
        
        l_id=[] 
        k=4
        max_i=0
        s=""	
        n=0
        for i, l in enumerate(labels):

            n=i+k
            s=str(n)				
            while ('1' in s) or('2' in s) or ('3' in s):				
                k+=1
                n=i+k
                s=str(n)											
            l_id.append(n)	
        label_list=list(zip(l_id, labels)) 
        '''      
        l_id=[] 
        k=4
        max_i=0
        s=""	
        n=0
	
        for char in lex:
            i=self.CHARMAP.index(char)	
            n= i+k
            s=str(n)				
            while ('1' in s) or('2' in s) or ('3' in s):				
                k+=1
                n=i+k
                s=str(n)											
            l_id.append(n)	
        label_list=list(zip(l_id, labels)) 
        '''         
        lex_new=[] 
        n=""
        c_idx=0
        n_idx=0		
        i=0
        for c in lex:
                if( c !='3') :	
                    n+=c				
                elif c=='3':
                    c_idx = int(n)
                    n+=c	
                    n_idx=int(n) 					
                    for i, labels in enumerate(label_list):
                         if c_idx == labels[0] :  						
                            lex_new.append(i+3)
                            n=""					
        #print ('lex_new, ', lex_new)
		
        return np.array(
           [GO_ID] + [i for i in lex_new] + [EOS_ID],
           dtype=np.int32)

 
        #return ([GO_ID]+[ int(lex) ]+[EOS_ID])

    @staticmethod
    def _parse_record(example_proto):
        features = tf.parse_single_example(
            example_proto,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string),
                'comment': tf.FixedLenFeature([], tf.string, default_value=''),
            })
        return features['image'], features['label'], features['comment']