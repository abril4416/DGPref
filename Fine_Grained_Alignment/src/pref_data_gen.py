import random
import argparse
from collections import defaultdict
from itertools import combinations
import os
import json
import pickle as pkl

import inflect
num_to_word_eng = inflect.engine()
import nltk
single = inflect.engine()
lemma = nltk.wordnet.WordNetLemmatizer()

from chair_utils import IMAGE_CONSTANT,FORBIDDEN_OBJ,get_mscoco_obj

def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data
    
def load_json(path):
    data=json.load(open(path,'rb'))
    return data

def count_module(check_items, det_info):
    check_info={}
    for item in check_items:
        num=item[0]
        obj=item[1]
        term=' '.join([str(num),obj])
        if len(det_info[obj]['scores'])==0:
            check_info[term]=-2
        else:
            if len(det_info[obj]['scores'])==num:
                check_info[term]=1
            else:
                check_info[term]=-1
    return check_info

def get_paired_data(scores,sample_num,diff_score):
    samples=[]
    for comp_idx1, comp_idx2 in combinations(scores, 2):
        score_1=comp_idx1["score"]
        score_2=comp_idx2["score"]
        score_diff=score_1-score_2
        if abs(score_diff)>=diff_score:
            if score_diff>0:
                prefer_idx=comp_idx1["idx"]
                reject_idx=comp_idx2["idx"]
                prefer=comp_idx1["caption"]
                reject=comp_idx2["caption"]
            elif score_diff<=0:
                prefer_idx=comp_idx2["idx"]
                reject_idx=comp_idx1["idx"]
                prefer=comp_idx2["caption"]
                reject=comp_idx1["caption"]
            else:
                continue
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
    """
    7/9/2024 Updates: 
        Average scores rather than number of wrong cases
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_num', type=int, 
                        default=3)
    parser.add_argument('--diff_score', type=float, 
                        default=0.3)#0.3 for object only, to control the size
    parser.add_argument('--raw_gen_file', type=str, 
                        default="../results/gen_results/raw_gen/llava-1.5-7b-hf/vsg.pkl")
    parser.add_argument('--expert_score_file', type=str, 
                        default="../../Decomp_Gen_Cap/decomp_results_llava_vsg_0.25/expert_check_scores.pkl")
    parser.add_argument('--PATH', type=str, 
                        default="/common/home/users/r/ruicao.2020/common2/Data_Storage/Rui_Data_Space/VQA")
    parser.add_argument('--DATA_NAME', type=str, 
                        default="vsg")
    parser.add_argument('--MODEL_NAME', type=str, 
                        default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument('--IMG_DIR_1', type=str, 
                        default="/PATH/VQA/VisualGenome/VG_100K")
    parser.add_argument('--IMG_DIR_2', type=str, 
                        default="/PATH/VQA/VisualGenome/VG_100K_2")
    """
    Consider all aspects as default
        can selectively use subset of aspects
        currently, simple summation
    """
    parser.add_argument('--CONSIDERED_ASPECTS', type=str, 
                        default="object,count,attribute,spatial,scale,text")
    """
    Whether using averaging expert scores
        averaging ==> score/num_valid
    """
    parser.add_argument('--AVG_ADV_SCORES', type=bool, 
                        default=True)
    args = parser.parse_args()

    prog_name_mapper={
        "obj_check":"object_result",
        "attribute_check":"attribute_result",
        "relation_check":"relation_result",
        "count_check":"count_result",
        "scale_check":"scale_result",
        "spatial_check":"spatial_result",
        "text_check":"text_result",
    }
    mscoco_objects, inverse_synonym_dict=get_mscoco_obj(args.PATH)
    print ('Length of coco objects:',len(mscoco_objects))
    model_name=args.MODEL_NAME.split('/')[-1]
    if args.DATA_NAME=='vsg':
        train_graphs=json.load(
            open(os.path.join(args.PATH,'GQA/train_sceneGraphs.json') ,'r'))
        names=list(train_graphs.keys())
        name_to_idx={name:i for i,name in enumerate(names)}
    elif args.DATA_NAME=='coco':
        img_dir=os.path.join(args.PATH,'VQA/train2014')
        names=os.listdir(img_dir)
    vsg_pred=load_pkl(os.path.join(args.raw_gen_file))
    expert_scores=load_pkl(args.expert_score_file)

    vis=0
    valid=0
    total={}
    select_scores={}
    hall_dict={}
    num_instances=0
    considered_aspects=args.CONSIDERED_ASPECTS.split(',')
    print ('What do we consider for preferred caption selection:',considered_aspects)
    for k,name in enumerate(names):
        if args.DATA_NAME=='vsg':
            idx=name_to_idx[name]
        elif args.DATA_NAME=='coco':
            idx=k
        if str(idx)+'_0' not in vsg_pred:
            continue
        if name not in expert_scores or len(expert_scores[name]["all_program"])==0:
            continue
        if vis%100==0:
            print (vis)
        #if vis>5:
        #    break
        #print (name)
        valid+=1
        eval_scores=expert_scores[name]["expert_scores"]
        all_programs=expert_scores[name]["all_program"]
        cap_scores=[]
        cap_avg_scores=[]
        for i in range(len(all_programs)):
            program=all_programs[i]
            per_program_score={prog_name_mapper[prog_name]:{} for prog_name in program}
            select_or_not={prog_name_mapper[prog_name]:{'score':0,'num_valid':0} for prog_name in program}
            #print (i)
            for prog_name in program:
                all_check_items=program[prog_name]
                #non_dup_items=defaultdict(int)
                #print (all_check_items)
                ref_info=eval_scores[prog_name_mapper[prog_name]]
                if prog_name in ["attribute_check","relation_check","text_check"]:
                    ref_info=ref_info['scores']
                elif prog_name == "count_check":
                    ref_info=count_module(all_check_items, eval_scores["object_result"])
                for check_item in all_check_items:
                    if prog_name == "obj_check":
                        if check_item in IMAGE_CONSTANT or check_item in FORBIDDEN_OBJ:
                            continue
                        """
                        Also need to try allowing the code below; mini batch=8
                        """
                        #if len(check_item.split(' '))>1 and check_item not in mscoco_objects:
                        #    continue
                        item=check_item
                        try:
                            norm_word=single.singular_noun(item.lower())
                        except:
                            norm_word=item.lower() 
                        if norm_word:
                            obj=norm_word
                        else:
                            obj=item.lower() 
                        if obj in ref_info:
                            ref_score=int(len(ref_info[obj]["scores"])>0)*2-1
                        else:
                            ref_score=int(len(ref_info[item]["scores"])>0)*2-1
                        #ref_score=int(len(ref_info[item]["scores"])>0)*2-1
                    elif prog_name=='count_check':
                        item=" ".join([str(check_item[0]),check_item[1]])
                        ref_score=ref_info[item]
                    elif prog_name=='text_check':
                        item=check_item
                        ref_score=ref_info[item]
                    else:
                        item="###".join([it for it in check_item])
                        if item not in ref_info:
                            #print (item)
                            continue
                        ref_score=ref_info[item]
                    per_program_score[prog_name_mapper[prog_name]][item]=ref_score
                cur_prog_scores=per_program_score[prog_name_mapper[prog_name]]
                for item in cur_prog_scores:
                    """
                    Update ==> Equally vs. Different scores
                    """
                    if cur_prog_scores[item]!=-2 and cur_prog_scores[item]<0:
                    #if cur_prog_scores[item]<0:
                        """
                        Number of wrong cases
                        """
                        select_or_not[prog_name_mapper[prog_name]]['score']+=cur_prog_scores[item]
                        """
                        if prog_name=='obj_check':
                            if valid<10:
                                print ('Doubel! Priority!')
                            select_or_not[prog_name_mapper[prog_name]]['score']+=cur_prog_scores[item]
                        """
                    if cur_prog_scores[item]!=-2 and cur_prog_scores[item]!=0: 
                    #if cur_prog_scores[item]!=0: 
                        select_or_not[prog_name_mapper[prog_name]]['num_valid']+=1
                        """
                        Updates: 18/9/24
                        """
                        #if cur_prog_scores[item]!=-2:
                        #    select_or_not[prog_name_mapper[prog_name]]['num_valid']+=1
                #print ('\t',prog_name_mapper[prog_name],per_program_score[prog_name_mapper[prog_name]])
                #print ('\t',select_or_not[prog_name_mapper[prog_name]])
                """
                As object score already indicate object hallucinations
                    ignore all -2 scores in attribute/relation related checks
                """
                
            total[name+'_'+str(i)]=per_program_score
            select_scores[name+'_'+str(i)]=select_or_not[prog_name_mapper[prog_name]]
            i_th_cap_score=0
            i_th_cap_score_avg=0.0
            i_th_cap_valid=0.0
            for asp in considered_aspects:
                if select_or_not[asp+'_result']['num_valid']:
                    i_th_cap_score_avg+=select_or_not[asp+'_result']['score']*1.0
                    i_th_cap_valid+=select_or_not[asp+'_result']['num_valid']
                i_th_cap_score+=select_or_not[asp+'_result']['score']
            cap_scores.append(i_th_cap_score)
            if i_th_cap_valid:
                i_th_cap_score_avg/=i_th_cap_valid
            cap_avg_scores.append(i_th_cap_score_avg)
            """
            print out details: non-trivial
                to make sure scores are in the same scale ==> better for choosing args.diff_score
            """
            #print ('\t',i,'-th CAP:',i_th_cap_score,'\tAVG:',i_th_cap_score_avg)
        if args.AVG_ADV_SCORES:
            sample_pair=get_paired_data([{
                "caption": vsg_pred[str(idx)+'_'+str(j)], "score":cap_avg_scores[j], "idx":j
            } for j in range(8)],args.sample_num,args.diff_score)
            if vis<10:
                print (vis)
                for pair in sample_pair:
                    pref_idx=pair["prefer_idx"]
                    rej_idx=pair["reject_idx"]
                    print ('Prefer score:',cap_avg_scores[pref_idx],'Reject score:',cap_avg_scores[rej_idx])
                print ('\n')
        else:
            sample_pair=get_paired_data([{
                "caption": vsg_pred[str(idx)+'_'+str(j)], "score":cap_scores[j], "idx":j
            } for j in range(8)],args.sample_num,args.diff_score)
        num_instances+=len(sample_pair)
        hall_dict[name]={
            "cap_scores":cap_scores,
            "cap_avg_scores":cap_avg_scores,
            "sample_pair":sample_pair
        }
        vis+=1
    pkl.dump(total,open('../pos_neg_pair_gen/process_score_file_details.pkl','wb'))
    pkl.dump(select_scores,open('../pos_neg_pair_gen/process_score_file_category.pkl','wb'))
    pkl.dump(hall_dict,
             open(os.path.join("../pos_neg_pair_gen",
                               '_'.join([args.DATA_NAME,model_name]),
                               'pos_neg_pairs-'+'_'.join(considered_aspects)+".pkl"),"wb"))
    print ('Number of valid:',valid)
    print ('\tNumber of instances:',num_instances)



