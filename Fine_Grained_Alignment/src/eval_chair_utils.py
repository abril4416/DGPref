import os
import torch
import numpy as np
import random
import json
import pickle as pkl
import config
from collections import defaultdict

from PIL import Image
from hal_cap_dataset import Hal_Cap_Data, load_json, load_pkl
from chair_utils import get_mscoco_obj, caption_to_words, get_node_obj_set

VQA_PATH="/mnt/data1/rui/Rui_Data_Space/VQA"
mscoco_objects, inverse_synonym_dict=get_mscoco_obj(VQA_PATH)
imid_to_objects=load_pkl(os.path.join(VQA_PATH,'Hal_Benchs/total_imgid_to_objects.pkl'))

def load_jsonl(path):
    total_info=[]
    with open(path,'rb')as f:
        d=f.readlines()
    for i,info in enumerate(d):
        data=json.loads(info)
        total_info.append(data)
    return total_info

def val_chair(pred):
    chair_i=0.0
    chair_s=0.0
    coco_word_count=0.0
    hall_word_count=0.0
    pred_obj_cat=0.0
    for idx in pred:
        gen_cap=pred[idx]
        gt_obj=imid_to_objects[int(idx)]
        words, node_words=caption_to_words(gen_cap, mscoco_objects, inverse_synonym_dict)
        num_pred_obj=len(node_words)
        num_hallucinated=len([obj for obj in node_words if obj not in gt_obj])
        hall_word_count+=num_hallucinated
        coco_word_count+=num_pred_obj
        if num_hallucinated>0:
            chair_s+=1
        pred_obj_cat+=len(set(node_words))
    chair_i=hall_word_count*100.0/coco_word_count
    chair_s=chair_s/len(pred)*100.0
    pred_obj_cat=pred_obj_cat/len(pred)
    print ('Chair i',chair_i)
    print ('Chair s',chair_s)
    print ('AVG obj pred',pred_obj_cat)
    return chair_s, chair_i, pred_obj_cat

if __name__=='__main__':

    pred_file_dir='../results/dpo_generation_llava1.5/object-count-attribute-spatial-scale-text'
    #pred_file_dir='../results/dpo_generation_llava1.5/filter'
    """
    all_files=os.listdir(pred_file_dir)
    for file in all_files:
        if file.startswith('amber')==False:
            continue
        print (file)
        all_pred=load_pkl(os.path.join(pred_file_dir,file))
        CHAIR , Cover, Ha_p, Ha=amber_eval(all_pred)
    """
    all_pred=load_pkl(os.path.join(pred_file_dir,'objhal_NUM_119_step_2000_bz_8.pkl'))
    #all_pred=load_pkl('../../RLAIF-V/RLAIF-V-7B-objhal.pkl')
    val_chair(all_pred)
    """
    all_files=os.listdir(pred_file_dir)
    for file in all_files:
        if file.startswith('objhal_NUM_23')==False:
            continue
        print (file)
        all_pred=load_pkl(os.path.join(pred_file_dir,file))
        val_chair(all_pred)
    """