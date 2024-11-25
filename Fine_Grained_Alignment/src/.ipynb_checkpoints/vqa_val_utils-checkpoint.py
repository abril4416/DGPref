import json
import os
import pickle as pkl
import random
from collections import defaultdict
import config
import numpy as np

import openai
from openai import OpenAI
os.environ['OPENAI_API_KEY']=open(
    '../../mine-RLHF/attr_rela_extraction/GPT_Key.txt','r').readlines()[0].strip()
client = OpenAI()

def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data

def load_json(path):
    data=json.load(open(path,'r'))
    return data

def load_jsonl(path):
    total_info=[]
    with open(path,'rb')as f:
        d=f.readlines()
    for i,info in enumerate(d):
        data=json.loads(info)
        total_info.append(data)
    return total_info

def get_eval(content, model_name, max_tokens):
    completion = client.chat.completions.create(
        model=model_name,
        messages=[{
            'role': 'system',
            'content': 'You are a helpful and precise assistant for checking the quality of the answer.'
        }, 
            {
                'role': 'user',
                'content': content,
            }],
        temperature=0.2,  # TODO: figure out which temperature is best for evaluation
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content


def parse_score(review):
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            print('error', review)
            return [-1, -1]
    except Exception as e:
        print(e)
        print('error', review)
        return [-1, -1]
    
def val_result_gpt(all_pred, model_name,save_file_name, args):
    question_file=load_jsonl(os.path.join(args.VQA_PATH,
                                          'llava-bench-in-the-wild/questions.jsonl'))
    context_file=load_jsonl(os.path.join(args.VQA_PATH,
                                         'llava-bench-in-the-wild/context.jsonl'))
    ref_file=load_jsonl(os.path.join(args.VQA_PATH,
                                     'llava-bench-in-the-wild/answers_gpt4.jsonl'))
    rule_dict=load_json(os.path.join(args.VQA_PATH,
                                'llava-bench-in-the-wild/rule.json'))
    image_to_context={context['image']: context for context in context_file}
    idx=0
    review_file = open('../results/gen_results/aux_gen/'+'_'.join(
        ["gpt_review",save_file_name,model_name])+'.jsonl', 'a')
    cur_reviews=[]
    for ques,ans1 in zip(question_file, ref_file):
        ques_id=ques['question_id']
        inst = image_to_context[ques['image']]
        if isinstance(inst['caption'], list):
            cap_str = '\n'.join(inst['caption'])
        else:
            cap_str = inst['caption']
        category = 'llava_bench_' + ques['category']
        if category in rule_dict:
            rule = rule_dict[category]
        else:
            assert False, f"Visual QA category not found in rule file: {category}."
        prompt = rule['prompt']
        role = rule['role']
        ans2=all_pred[ques_id]
        content = (f'[Context]\n{cap_str}\n\n'
                   f'[Question]\n{ques["text"]}\n\n'
                   f'[{role} 1]\n{ans1["text"]}\n\n[End of {role} 1]\n\n'
                   f'[{role} 2]\n{ans2}\n\n[End of {role} 2]\n\n'
                   f'[System]\n{prompt}\n\n')
        cur_js = {
            'id': idx+1,
            'question_id': ques_id,
            'answer1_id': ans1.get('answer_id', ans1['question_id']),
            'answer2_id': ques_id,
            'category': category
        }
        if idx >= len(cur_reviews):
            review = get_eval(content, model_name, 1024)
            scores = parse_score(review)
            cur_js['content'] = review
            cur_js['tuple'] = scores
            review_file.write(json.dumps(cur_js) + '\n')
            review_file.flush()
        else:
            print(f'Skipping {idx} as we already have it.')
        idx += 1
        print(idx)
    return 

"""
Evaluation codes for VQA datasets:
    VizWiz, TextQA, GQA, LLaVA-in-the-wild
For llava in the wild
    call gpt-4
"""
def val_vqa_results(args, all_pred, dataset_name, data):
    if dataset_name=='llava-bench':
        acc=val_result_gpt(all_pred, args)
        return acc
    acc=0.0
    names=list(all_pred.keys())
    #print (names[0:10])
    for i,row in enumerate(data):
        if dataset_name=='gqa':
            idx=str(row['image_id'])
            answers={row['answer']:10}
        elif dataset_name=='text-vqa':
            idx=str(i)
            answers=defaultdict(int)
            #print (row)
            for ans in row['answers']:
                answers[ans]+=1
        elif dataset_name=='vizwiz':
            idx=str(i)
            answers=defaultdict(int)
            for info in row['answers']:
                answers[info['answer']]+=1
        #idx=str(i)
        pred=all_pred[idx].lower().strip()
        #if dataset_name=='vizwiz':
        #    print (pred,answers)
        if pred in answers:
            ans_count=answers[pred]
            acc+=min(1.0,ans_count/3.0)
    acc=acc*100.0/len(all_pred)
    print ('Length of dataset:',len(all_pred))
    print ('\tAccuracy for dataset %s is %.2f' % (dataset_name,acc))
    return acc

def summarize_gpt_review(review_file):
    scores = defaultdict(list)
    for review in review_file:
        if 'category' in review:
            scores[review['category']].append(review['tuple'])
            scores['all'].append(review['tuple'])
        else:
            if 'tuple' in review:
                scores['all'].append(review['tuple'])
            else:
                scores['all'].append(review['score'])
    for k, v in sorted(scores.items()):
        stats = np.asarray(v).mean(0).tolist()
        stats = [round(x, 3) for x in stats]
        print(k, round(stats[1]/stats[0]*100, 1), round(stats[0] * 10, 1), round(stats[1] * 10, 1))
    print('=================================')
    return
    
if __name__=='__main__':
    pred_file_dir='../results/dpo_generation_llava1.5/object-count-attribute-spatial-scale-text'
    #pred_file_dir="../results/gen_results/raw_gen/llava-1.5-7b-hf"
    import config
    args=config.parse_opt()
    #pred_file_dir='../results/gen_results/raw_gen/llava-1.5-7b-hf'
    datasets=['llava-bench']
    """
        text-vqa question id is not unique!
    """
    for dataset_name in datasets:
        print (dataset_name)
        if dataset_name=='gqa':
            gqa_data=load_json(os.path.join(args.VQA_PATH,'GQA','testdev_balanced_questions.json'))
            data=[]
            for idx in gqa_data:
                gqa_row=gqa_data[idx]
                gqa_row['image_id']=idx#using the key as the unique indexing
                data.append(gqa_row)
        elif dataset_name=='text-vqa':
            data=load_jsonl(os.path.join(args.VQA_PATH,'text-vqa/llava_textvqa_val_v051_ocr.jsonl'))
        elif dataset_name=='vizwiz':
            data=load_json(os.path.join(args.VQA_PATH,'vizwiz/val.json'))
        else:
            data=load_jsonl(os.path.join(args.VQA_PATH,'llava-bench-in-the-wild/questions.jsonl'))
        #pred_file=load_pkl(os.path.join(pred_file_dir,dataset_name+'.pkl'))
        #pred_file=load_pkl(os.path.join(pred_file_dir,dataset_name+'_NUM_26_step_40000_bz_8.pkl'))
        """
        all_pred={}
        for i,row in enumerate(data):
            pred=pred_file[str(i)]
            if dataset_name=='gqa':
                idx=str(row['image_id'])
            elif dataset_name=='text-vqa':
                idx=str(i)
            elif dataset_name=='vizwiz':
                idx=str(i)
            elif dataset_name=='llava-bench':
                idx=row['question_id']
            all_pred[idx]=pred
        """
        if dataset_name=='text-vqa':
            data=load_json(os.path.join(args.VQA_PATH,'text-vqa/val.json'))["data"]
        save_file_name=dataset_name+'_NUM_0_step_0_bz_8.pkl'
        #save_file_name="llava-bench.pkl"
        all_pred= load_pkl(os.path.join(pred_file_dir,save_file_name))
        #if dataset_name=='gqa':
        #    print (all_pred)
        print ('Length of dataset:',len(all_pred))
        if dataset_name=='llava-bench':
            model_name='gpt-4-0613'
            #model_name="gpt-3.5-turbo-0125"
            model_name="gpt-4o"
            if os.path.exists(
                '../results/gen_results/aux_gen/'+'_'.join(["gpt_review",save_file_name.split('.')[0],model_name])+'.jsonl'
            )==False:
                print ('Generating reviews first!')
                val_result_gpt(all_pred, model_name, save_file_name.split('.')[0], args)
            review_file=load_jsonl('../results/gen_results/aux_gen/'+'_'.join(["gpt_review",save_file_name.split('.')[0],model_name])+'.jsonl')
            summarize_gpt_review(review_file)
        else:
            acc=val_vqa_results(args, all_pred, dataset_name,data)