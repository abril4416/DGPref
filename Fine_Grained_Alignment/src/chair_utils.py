import random
import argparse
from collections import defaultdict

import os
import json
import pickle as pkl

import nltk
import inflect
from itertools import combinations
single = inflect.engine()
lemma = nltk.wordnet.WordNetLemmatizer()

IMAGE_CONSTANT=['scene','image','picture','photo','view','frame','figure','side','backdrop','background']
FORBIDDEN_OBJ=['area','field','ground','grass','glass',
 'checkpoint','city','town','other person', 'precision', 'object','texture','speed','focus', 'right side', 'subject', 'object','remote','side', 'body of water','individual', 'foreground', 'atmosphere', 'waterfront', 'waterside','step','figure','air','skill', 'wall', 'setting','jumping', 'city street', 'side of street', 'surface', 'winter sport','slope','snow-covered slope', 'backdrop','edge of slope','edge','sun', 'winter','design','space', 'detail','arch', 'side of building', 'landmark','element','color','landscape','group','filled', 'with each other','event', 'array','room', 'around table', 'gathering','center of attention','around', 'further away','familiy','size', 'environment', 'above', 'nearby', 'center', 'center of image', 'viewer', 'ambiance','light', 'mode of transportation','transportation','position','direction','assortment', 'center of scene', 'turn', 'each other', 'pattern']

def get_node_obj_set(obj_list,mscoco_objects,inverse_synonym_dict):
    words=[]
    for w in obj_list:
        norm_word=single.singular_noun(w)
        if norm_word==False:
            words.append(w)
        else:
            words.append(norm_word)
    if ('toilet' in words) & ('seat' in words): words =\
        [word for word in words if word != 'seat']
    
    #get synonyms for all words in the GT obj set
    words = [word for word in words if word in set(mscoco_objects)]
    node_words = []
    for word in words:
        node_words.append(inverse_synonym_dict[word])
    node_list=set(node_words)
    return node_list

def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data
    
def load_json(path):
    data=json.load(open(path,'rb'))
    return data

def get_mscoco_obj(path):
    synonyms = open(os.path.join(path,'Hal_Benchs/synonyms.txt')).readlines()
    synonyms = [s.strip().split(', ') for s in synonyms]
    ms_coco_objects=[]
    inverse_synonym_dict = {}
    for synonym in synonyms:
        ms_coco_objects.extend(synonym)#node object, obj_1, obj_2 ...
        for s in synonym:
            inverse_synonym_dict[s] = synonym[0]
    coco_double_words = ['motor bike', 'motor cycle', 'air plane', 
                         'traffic light', 'street light', 'traffic signal',
                         'stop light', 'fire hydrant', 'stop sign',
                         'parking meter', 'suit case', 'sports ball',
                         'baseball bat', 'baseball glove', 'tennis racket', 
                         'wine glass', 'hot dog', 'cell phone', 'mobile phone',
                         'teddy bear', 'hair drier', 'potted plant', 'bow tie', 
                         'laptop computer', 'stove top oven', 'hot dog',
                         'teddy bear', 'home plate', 'train track']
    animal_words = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
                    'elephant', 'bear', 'zebra', 'giraffe', 'animal', 'cub']
    vehicle_words = ['jet', 'train']
    for double_word in coco_double_words:
        ms_coco_objects.append(double_word)
        inverse_synonym_dict[double_word]=double_word
    for animal_word in animal_words:
        ms_coco_objects.append(animal_word)
        inverse_synonym_dict[animal_word]=animal_word
        real_word="baby "+animal_word
        ms_coco_objects.append(real_word)
        inverse_synonym_dict[real_word]=animal_word
        real_word="adult "+animal_word
        ms_coco_objects.append(real_word)
        inverse_synonym_dict[real_word]=animal_word
    for vehicle_word in vehicle_words:
        inverse_synonym_dict[vehicle_word]=vehicle_word
        ms_coco_objects.append(vehicle_word)
        real_word="passenger "+vehicle_word
        ms_coco_objects.append(real_word)
        inverse_synonym_dict[real_word]=vehicle_word
        
    ms_coco_objects.append("bow tie")
    inverse_synonym_dict["bow tie"]="tie"
    ms_coco_objects.append("toilet seat")
    inverse_synonym_dict["toilet seat"]="toilet"
    ms_coco_objects.append("wine glas")
    inverse_synonym_dict["wine glas"]="wine glass"
    return ms_coco_objects, inverse_synonym_dict

def caption_to_words(caption, mscoco_objects, inverse_synonym_dict):
    #standard preprocessing
    
    nltk_words = nltk.word_tokenize(caption.lower())
    words = []
    for w in nltk_words:
        norm_word=single.singular_noun(w)
        if norm_word==False:
            words.append(w)
        else:
            words.append(norm_word)
    """
    words = nltk.word_tokenize(caption.lower())
    words_2 = [lemma.lemmatize(w) for w in words]
    words = words_2
    """
    #replace double words
    i = 0
    
    if ('toilet' in words) & ('seat' in words): words =\
        [word for word in words if word != 'seat']
    words = [word for word in words if word in set(mscoco_objects)]
    node_words = []
    for word in words:
        node_words.append(inverse_synonym_dict[word])
    #return all the MSCOCO objects in the caption
    #node words for the root object
    return words, node_words

def get_paired_data(cur_hall,sample_num,diff_score):
    samples=[]
    for comp_idx1, comp_idx2 in combinations(cur_hall, 2):
        score_1=comp_idx1["num_hall"]
        score_2=comp_idx2["num_hall"]
        score_diff=score_1-score_2
        if abs(score_diff)>=diff_score:
            if score_diff<0:
                prefer_idx=comp_idx1["idx"]
                reject_idx=comp_idx2["idx"]
                prefer=comp_idx1["caption"]
                reject=comp_idx2["caption"]
            else:
                prefer_idx=comp_idx2["idx"]
                reject_idx=comp_idx1["idx"]
                prefer=comp_idx2["caption"]
                reject=comp_idx1["caption"]
        else:
            continue
        samples.append({
            "prefer":prefer,
            "reject":reject,
            "prefer_idx":prefer_idx,
            "reject_idx":reject_idx
        })
    return samples

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_num', type=int, 
                        default=3)
    parser.add_argument('--diff_score', type=int, 
                        default=2)
    parser.add_argument('--raw_gen_file', type=str, 
                        default="../raw_generation_llava1.5/vsg_train.pkl")
    parser.add_argument('--PATH', type=str, 
                        default="/common2/ruicao.2020/Data_Storage/Rui_Data_Space/VQA")
    parser.add_argument('--IMG_DIR_1', type=str, 
                        default="/common2/ruicao.2020/Data_Storage/Rui_Data_Space/VQA/VisualGenome/VG_100K")
    parser.add_argument('--IMG_DIR_2', type=str, 
                        default="/common2/ruicao.2020/Data_Storage/Rui_Data_Space/VQA/VisualGenome/VG_100K_2")
    args = parser.parse_args()

    #loading pred training data
    gen_detail_cap=load_pkl(args.raw_gen_file)
    
    """
    Loading GT graphs
        1) scores from object hallucinations only
    """
    train_graphs=load_json(
        os.path.join(args.PATH,'GQA/train_sceneGraphs.json')
        )
    print (len(train_graphs))
    mscoco_objects, inverse_synonym_dict=get_mscoco_obj(args.PATH)
    print ('Length of the coco object list:',len(mscoco_objects))

    names=list(train_graphs.keys())
    print (len(names),len(gen_detail_cap)//8)
    name_to_idx={name:i for i,name in enumerate(names)}
    hall_dict={}
    for name in names:
        idx=name_to_idx[name]
        if str(idx)+'_0' not in gen_detail_cap:
            continue
            
        info=train_graphs[str(name)]
        width=info['width']
        height=info['height']
        obj_infos=info['objects']
        ids_to_names={}
        img_obj=defaultdict(int)

        for obj_idx in obj_infos:
            obj_info=obj_infos[obj_idx]
            obj_name=obj_info['name']
            ids_to_names[obj_idx]=obj_name
        for obj_idx in obj_infos:
            obj_info=obj_infos[obj_idx]
            obj_name=obj_info['name']
            img_obj[obj_name]+=1
            
        gt_obj=[o for o in img_obj]
        gt_obj=get_node_obj_set(gt_obj, mscoco_objects, inverse_synonym_dict)
        
        cur_hall=[]
        for k in range(8):
            cap=gen_detail_cap[str(idx)+'_'+str(k)]
            words, node_words=caption_to_words(cap, mscoco_objects, inverse_synonym_dict)
            num_pred_obj=len(node_words)
            word_dict_0=defaultdict(int)
            for w in node_words:
                word_dict_0[w]+=1
            word_list_0=[w for w in word_dict_0]
            num_hallucinated=len([obj for obj in word_list_0 if obj not in gt_obj])
            cur_hall.append({
                "caption":cap,
                "idx":k,
                "num_hall":num_hallucinated,
                "num_pred_obj":len(words)
            })
            #print ('For ',k,'-th caption, number of hallucinated objects:',len([obj for obj in word_list_0 if obj not in gt_obj]))
            #print ('\tNumber of predicted objects:',len(words))
        sample_pair=get_paired_data(cur_hall,args.sample_num,args.diff_score)
        hall_dict[name]={
            "obj_pred_info":cur_hall,
            "sample_pair":sample_pair
        }
        """
        for pair in sample_pair:
            print (pair["prefer_idx"],pair["reject_idx"])
            print ('\t',cur_hall[pair["prefer_idx"]],cur_hall[pair["reject_idx"]])
        """    
    pkl.dump(hall_dict,open("pos_neg_pairs.pkl","wb"))
    

