import numpy as np
import argparse
import json
import nltk
import tqdm
import os
import pickle as pkl
import random
from collections import defaultdict
from PIL import Image

from utils import process_decomp_results
from constant import SPATIAL_RELA, SCALE, FORBIDDEN_ATTR, IMAGE_CONSTANT,FORBIDDEN_OBJ
import module_config
import inflect
from modules import ExpertModules
num_to_word_eng = inflect.engine()
all_forbid_obj=[]
all_forbid_obj.extend(IMAGE_CONSTANT)
all_forbid_obj.extend(FORBIDDEN_OBJ)

import nltk
from itertools import combinations
single = inflect.engine()
lemma = nltk.wordnet.WordNetLemmatizer()

def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data

def load_json(path):
    data=json.load(open(path,'r'))
    return data

def get_mapper():
    count_mapper={num_to_word_eng.number_to_words(i):i for i in range(2,21)}
    rela_mapper={
        "left":"left","right":"right",
        "top":"top","above":"top",
        "bottom":"bottom","below":"bottom","under":"bottom","beneath":"bottom","underneath":"bottom",
        "near":"near","next":"near"
    }
    scale_mapper={
        "large":"large", "huge":"large","larger":"large","big":"large",
        "small":"small","tiny":"small","smaller":"small",
        "long":"long","short":"short",
        "tall":"tall","high":"tall"
    }
    return count_mapper, rela_mapper, scale_mapper
    
def gen_program(decomp_results,count_mapper, rela_mapper, scale_mapper):
    """
    De-duplication ==> One image, decompositional results may be duplicated
        e.g., need to check the existent of the same objects/relations between the same objects
        Although dependecies in a program (e.g., check "red car" calls for checking "car" first)
            Here we just try the best to make it parellel to be more efficient
    Aspects: attribute, entity, relation, count, text
    Aggregate all objects for detection
    Counting heuristics
    Relation heuristics
    """
    det_obj=defaultdict(int)
    count_prog=[]
    sp_prog=[]
    rela_prog=[]
    attr_prog=[]
    scale_prog=[]
    for aspect in decomp_results:
        if aspect=='entity':
            for obj in decomp_results[aspect]:
                det_obj[obj]+=1
        elif aspect=='attribute':
            for info in decomp_results[aspect]:
                if info[1] in IMAGE_CONSTANT:
                    continue
                det_obj[info[1]]+=1
                attr=info[0]
                if attr in FORBIDDEN_ATTR:
                    continue
                if attr in scale_mapper:
                    map_attr=scale_mapper[attr]
                    scale_prog.append((map_attr,info[1]))
                else:
                    attr_prog.append(info)
        elif aspect=='relation':
            for info in decomp_results[aspect]:
                if info[0] in IMAGE_CONSTANT:
                    continue
                det_obj[info[0]]+=1
                det_obj[info[-1]]+=1
                rela=info[1]
                map_flag=False
                for norm_rela in rela_mapper:
                    if norm_rela in rela:
                        map_rela=rela_mapper[norm_rela]
                        map_flag=True
                        break
                if map_flag:
                    sp_prog.append((info[0],map_rela,info[-1]))
                else:
                    rela_prog.append(info)
        elif aspect=='count':
            for info in decomp_results[aspect]:
                if info[1] in IMAGE_CONSTANT:
                    continue
                det_obj[info[1]]+=1
                num=count_mapper[info[0]]
                count_prog.append((num, info[1]))
    det_obj=[o for o in det_obj.keys() if o not in IMAGE_CONSTANT]      
    program={
        'obj_check':det_obj, #be careful, obj_check already includes objects in count, relation and attributes
        'count_check':count_prog,
        'attribute_check':attr_prog,
        'relation_check':rela_prog,
        'scale_check':scale_prog,
        'spatial_check':sp_prog,
        'text_check':decomp_results['text']
    }
    return program


def gen_parallel_program(all_programs):
    #print (len(all_programs))
    prog_names=all_programs[0].keys()
    prog_arguments={prog_name:defaultdict(int) for prog_name in prog_names}
    for i in range(len(all_programs)):
        program=all_programs[i]
        for name in prog_names:
            if name in ["obj_check", "text_check"]:
                for obj in program[name]:
                    prog_arguments[name][obj]+=1
            elif name in ['spatial_check','attribute_check','relation_check','scale_check']:
                for obj in program[name]:
                    prog_arguments[name]['###'.join([arg for arg in obj])]+=1
    parallel_program={prog_name: list(prog_arguments[prog_name].keys()) for prog_name in prog_names}
    return parallel_program
    
if __name__=='__main__':
    """
    warning!!!
    replace with your own file path
    """
    GQA_PATH="/PATH/VQA/GQA"
    IMG_DIR_1='/PATH/VQA/VisualGenome/VG_100K'
    IMG_DIR_2='/PATH/VQA/VisualGenome/VG_100K_2'
    
    train_graphs=json.load(
        open(os.path.join(GQA_PATH,'train_sceneGraphs.json'),'r'))
    names=list(train_graphs.keys())
    name_to_idx={name:i for i,name in enumerate(names)}
    ext_file=load_pkl('../decomp_results_llava_vsg_0.25/all_vsg_llama3.pkl')
    """
    warning!!!
    replace with your own path for base MLLM generated detailed image descriptions
    """
    vsg_pred=load_pkl(YOUR_CAP_PATH)
    print ('Decomp result: ',len(ext_file),'Raw prediction: ',len(vsg_pred))
    
    aspects=["attribute","count","entity","relation","text"]
    count_mapper, rela_mapper, scale_mapper=get_mapper()
    module_args=module_config.parse_opt()
    experts=ExpertModules(module_args, module_args.DEBUG)
    
    vis=0
    total={}
    total=load_pkl('../decomp_results_llava_vsg_0.25/expert_check_scores.pkl')
    print ('Already generated scores:',len(total))
    random.shuffle(names)
    for name in names:
        if module_args.DEBUG:
            if vis>15:
                break
        if name in total and len(total[name]["all_program"])>0:
            continue
        idx=name_to_idx[name]
        if str(idx)+'_0' not in vsg_pred:
            continue
        flag=False
        all_programs=[]
        img_id=str(name)+'.jpg'
        total[name]={}
        if os.path.exists(os.path.join(IMG_DIR_1,img_id)):
            img_path=os.path.join(IMG_DIR_1,img_id)
        elif os.path.exists(os.path.join(IMG_DIR_2,img_id)):
            img_path=os.path.join(IMG_DIR_2,img_id)
        image=Image.open(img_path).convert('RGB')
        for i in range(8):
            if str(idx)+'_'+str(i) not in ext_file:
                continue
            pred=vsg_pred[str(idx)+'_'+str(i)]
            ext_info=ext_file[str(idx)+'_'+str(i)]
            process_ext_result={'program':{},'decomp_result':{}}
            for aspect in aspects:
                process_info=process_decomp_results(pred, ext_info, aspect)
                process_ext_result['decomp_result'][aspect]=process_info
                #print ('\t',aspect,'\n\t\t',process_info)
            flag=True
            program=gen_program(process_ext_result['decomp_result'],count_mapper,rela_mapper,scale_mapper)
            process_ext_result['program']=program
            """
            for prog_name in program:
                print (prog_name)
                print ('\t',program[prog_name])
            """
            all_programs.append(program)
        total[name]["image_path"]=img_path
        total[name]["all_program"]=all_programs
        """
        highly duplicated checking items
        make them parallel ==> gen_parallel_program
        """
        if flag==False:
            continue
        parallel_modules=gen_parallel_program(all_programs)
        if module_args.DEBUG:
            print (img_path)
            for prog_name in parallel_modules:
                print (prog_name)
                print ('\t',parallel_modules[prog_name])
        det_info, obj_result=experts.obj_module(image, parallel_modules["obj_check"])
        attribute_result=experts.vqa_module(image, parallel_modules["attribute_check"],"attribute_check",det_info,validate_ques=True)
        relation_result=experts.vqa_module(image, parallel_modules["relation_check"],"relation_check",det_info,validate_ques=True)
        count_result=experts.count_module(parallel_modules["count_check"], det_info)
        scale_result=experts.scale_module(image, parallel_modules["scale_check"], det_info)
        spatial_result=experts.spatial_module(image, parallel_modules["spatial_check"], det_info)
        if len(parallel_modules["text_check"])>0:
            text_result=experts.text_module(img_path, parallel_modules["text_check"])
        else:
            text_result={"details":[],"scores":{},"det_texts":[]}
        if module_args.DEBUG:
            print (img_path)
            print ('Object:\n\t',obj_result)
            print ('Attribute:\n\t',attribute_result)
            print ('Relation:\n\t',relation_result)
            print ('Count:\n\t',count_result)
            print ('Scale:\n\t',scale_result)
            print ('Spatial:\n\t',spatial_result)
            print ('Text:\n\t',text_result)        
        all_checks={
            "object_result":det_info,
            "attribute_result":attribute_result,
            "relation_result":relation_result,
            "count_result":count_result,
            "scale_result":scale_result,
            "spatial_result":spatial_result,
            "text_result":text_result
        }
        total[name]["expert_scores"]=all_checks
        if flag:
            vis+=1
        if vis%50==0 and module_args.DEBUG==False:
            pkl.dump(total,open('../decomp_results_llava_vsg_0.25/expert_check_scores.pkl','wb'))
            print ('Already finished:',vis)
            print (img_id, img_path)
    pkl.dump(total,open('../decomp_results_llava_vsg_0.25/expert_check_scores.pkl','wb'))
            