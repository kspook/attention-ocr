#!C:\Users\60067527\Anaconda3\envs\py36
#-*- coding: utf-8 -*-
from __future__ import absolute_import
import os, io
import logging
import re

import tensorflow as tf

from six import b
import numpy as np



SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH,
                                  #'../labels/bank_labelsSW.txt')
                                  '../labels/bank_labels.txt')

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def generate(annotations_path, output_path, log_step=5000,
             force_uppercase=True, save_filename=False):

    logging.info('Building a dataset from %s.', annotations_path)
    logging.info('Output file: %s', output_path)

    writer = tf.python_io.TFRecordWriter(output_path)
    longest_label = ''
    idx = 0

		
    with open(annotations_path, 'r', encoding='utf-8') as annotations:
        word=[]	    
      
        for idx, line in enumerate(annotations):
            line = line.rstrip('\n')

            # Split the line on the first whitespace character and allow empty values for the label
            # NOTE: this does not allow whitespace in image paths
            line_match = re.match(r'(\S+)\s(.*)', line)
            #print('line  ', line)			
            if line_match is None:
                logging.error('missing filename or label, ignoring line %i: %s', idx+1, line)
                continue
            (img_path, label) = line_match.groups()
            #print(img_path, label)			

            with open(img_path, 'rb') as img_file:
                img = img_file.read()

#            if force_uppercase:
#                label = label.upper()
      
            try:
                word= convert_lex(label)
 
            except IOError:
                    pass # ignore error images		

            if len(label) > len(longest_label):
                longest_label = label
				
            '''            
            feature = {}
            feature['image'] = _bytes_feature(img)
            feature['label'] = _bytes_feature(b(label))
            '''
            label = word
            label=''.join(map(str,label))
			
            feature = {}
            feature['image'] = _bytes_feature(img)
            feature['label'] = _bytes_feature(b(label))
  
 			
            if save_filename:
                feature['comment'] = _bytes_feature(b(img_path))

            example = tf.train.Example(features=tf.train.Features(feature=feature))

            writer.write(example.SerializeToString())

            if idx % log_step == 0:
                logging.info('Processed %s pairs.', idx+1)
				

				
    if idx:			
        logging.info('Dataset is ready: %i pairs.', idx+1)
        logging.info('Longest label (%i): %s', len(longest_label), longest_label)

    writer.close()

def convert_lex( lex):

        #if sys.version_info >= (3,):
        #    lex = lex.decode('utf-8')
            #lex = lex.decode('iso-8859-1')
        
        #assert len(lex) < self.bucket_specs[-1][1]
		#return np.array(
        #    [self.GO_ID] + [self.CHARMAP.index(char) for char in lex] + [self.EOS_ID],
        #    dtype=np.int32)

    GO_ID = 1
    EOS_ID = 2
    CHR_BR = 3
	
    label_file = DEFAULT_LABEL_FILE
    with io.open(label_file, 'r', encoding='utf-8') as f:
        labels = f.read().splitlines()
    #print(labels)
     	
    l_id=[] 
    k=3
    s=""	
    n=0
    for i, l in enumerate(labels):
        n=i+k
        s=str(n)				
        #print('i  l k n s ', i , l, k, n, s)
        while ('1' in s) or('2' in s) or ('3' in s):				
                k+=1
                n=i+k
                s=str(n)											
                #print('while  i l n k s: ',  i, l , k, n, s)
        l_id.append(n)	
        #print('i  l k n s l_id', i , l, k, n, s, l_id)		

        label_list=list(zip( (j for j in range(0,i+1)),l_id, labels)) 
    #print('label_list, ' , label_list)
	
    lex_new=[] 
    j=0
    for c in lex:
        #print('c ord(c) lex', c, ord(c), lex)
        for j, l_id, label in label_list:		
        #for i, l in enumerate(labels):
            #print('c j  l_id label', c, j , l_id, label)	
            if c == label:			
                 lex_new.append(l_id)
                 lex_new.append(3)				 
 
 	
    return lex_new
    ''' 
    return np.array(
       #[i for i in lex_new],
       [i for i in lex_new],	   
       #[GO_ID] + [i for i in lex_new] + [EOS_ID],	  
       #[GO_ID]  + [EOS_ID],	  	   
       dtype=np.int32)
	'''   