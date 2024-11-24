import numpy as np
import argparse
import json
import nltk
import tqdm
import os
import pickle as pkl
import random
import inflect
import re
def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data

def load_json(path):
    data=json.load(open(path,'r'))
    return data

valid_num_parts={
    'entity':1,
    'attribute':2,
    'count':2,
    'relation':3
}
num_to_word_eng = inflect.engine()
count_mapper={str(i):num_to_word_eng.number_to_words(i) for i in range(2,21)}
valid_counts=[num_to_word_eng.number_to_words(i) for i in range(2,21)]

def det_ocr_text(str):
    ocr_text=re.findall(r'"(.*?)"',str)
    return ocr_text

def extract_brackets(str):
    all_info=re.findall(r'\(.*?\)',str) # each in format of  (xx,xxx, xxx...)
    return all_info

def process_decomp_results(raw_pred, ext_info, str_type):
    if str_type in ['attribute','count','relation']:
        str=ext_info[str_type]
        all_info=extract_brackets(str)
        valid_len=valid_num_parts[str_type]
        process_info=[]
        for info in all_info:
            info=info[1:-1]
            splits=info.split(', ')
            if len(splits)!=valid_len:
                continue
            if str_type=='count':
                if splits[0].isdigit() and splits[0] in count_mapper:
                    count_num=count_mapper[splits[0]]
                else:
                    count_num=splits[0]
                if count_num not in valid_counts:
                    continue
                cur_info=(count_num, splits[1])
            elif str_type=='attribute':
                if splits[0].strip()==splits[1].strip():
                    continue
                cur_info=(splits[0].strip(),splits[1].strip())
            elif str_type=='relation':
                cur_info=(splits[0].strip(),splits[1].strip(),splits[2].strip())
            process_info.append(cur_info)
    elif str_type == 'text':
        process_info=det_ocr_text(raw_pred)
    elif str_type=='entity':
        str=ext_info[str_type]
        all_info=extract_brackets(str)
        if len(all_info)==0:
            all_info=str
            process_info_raw=all_info.split(', ')
        else:
            all_info=all_info[0][1:-1]
            process_info_raw=all_info.split(', ')
        process_info=[p.strip() for p in process_info_raw]
    return process_info
