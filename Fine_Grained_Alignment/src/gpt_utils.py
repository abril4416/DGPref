import json
import os
import pickle as pkl
import random
from collections import defaultdict
import config
import time
import openai
from openai import OpenAI
os.environ['OPENAI_API_KEY']=YOUR_API_KEY
client = OpenAI()

def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data

def load_json(path):
    data=json.load(open(path,'r'))
    return data

args=config.parse_opt()

def gpt_eval(all_pred,model_name,file_name):
    template=open("eval_templates/gpt_template.txt","r").readlines()
    template='\n'.join(template)
    names=list(all_pred.keys())
    annotation_file=load_json("/mnt/data1/rui/Rui_Data_Space/VQA/Hal_Benchs/mmhal-bench_answer_template.json")
    gt_annotation={}
    for row in annotation_file:
        img_id=row["image_id"]
        gt_annotation[img_id]={
            "question_topic":row["question_topic"],
            "image_content":row["image_content"],
            "question":row["question"],
            "gt_answer":row["gt_answer"]
        }
    scores=[]
    hallucinations=[]
    all_rsp=[]
    print ('Evaluation using:',model_name)
    for i,name in enumerate(names):
        pred=all_pred[name]
        record=gt_annotation[name]
        image_content = ', '.join(record['image_content'])
        input_text = template.format(image_content, record['question'], record['gt_answer'], pred)
        completion = client.chat.completions.create(
            #model="gpt-3.5-turbo-0125",
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text}
                ]
            )
        response=completion.choices[0].message.content
        all_rsp.append(response)
        scores_found = []
        for s in range(7):
            if f'rating: {s}' in response.lower():
                scores_found.append(s)
        if len(scores_found) == 1:
            scores.append(scores_found[0])
        else:
            print('Warning: multiple or zero scores found')
            print(i, response)
            scores.append(0)
        if scores[-1] >= 3:
            hallucinations.append(0)
        else:
            hallucinations.append(1)
        """
        print (i)
        print ('\t',pred)
        print (response)
        """
    print('Average score: {:.2f}'.format(sum(scores) / len(scores)))
    print('Hallucination rate: {:.2f}'.format(sum(hallucinations) / len(hallucinations)))
    json.dump(all_rsp,open('../results/gen_results/aux_gen/'+'_'.join(["gpt_review",file_name,model_name])+'.json','w'))
    return sum(scores)*1.0 / len(scores), sum(hallucinations)*1.0 / len(hallucinations)

if __name__=='__main__':
    pred_file_dir='../results/dpo_generation_llava1.5/object-object-count-attribute-spatial-scale-text'
    model_name='gpt-4-0613'
    
    file_name="mmhal_NUM_113_step_4800_bz_8.pkl"
    print(file_name)
    all_pred=load_pkl(os.path.join(pred_file_dir,file_name))
    gpt_eval(all_pred,model_name,file_name.split('.')[0])