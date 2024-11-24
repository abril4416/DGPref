import numpy as np
import argparse
import json
import nltk
import tqdm
import os
import pickle as pkl
import random
import torch
import transformers

def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data

def load_json(path):
    data=json.load(open(path,'r'))
    return data
    
parser = argparse.ArgumentParser(description='Generate extra questions based on claims with a prompt. Useful for searching.')
parser.add_argument('--GQA_PATH', 
                    default="", 
                    )
parser.add_argument('--target_file', default="")
args = parser.parse_args()

train_graphs=json.load(
    open(os.path.join(args.GQA_PATH,'train_sceneGraphs.json')
         ,'r')
)
print (len(train_graphs))

"""
For templates, each provide eight demonstration
Categories:
    Entities (people, animals, objects)
    Attributes (color, activity, shape, material, scale)
    Relations 
    Counting numbers
    OCR texts
"""
IMG_DIR_1='/PATH/VQA/VisualGenome/VG_100K'
IMG_DIR_2='/PATH/VQA/VisualGenome/VG_100K_2'
"""
warning!!!
replace with your own file path
"""
names=list(train_graphs.keys())
name_to_idx={name:i for i,name in enumerate(names)}

print (torch.__version__)
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
  "text-generation",
  model="meta-llama/Meta-Llama-3-8B-Instruct",
  model_kwargs={"torch_dtype": torch.bfloat16},
  device="cuda",
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
"""
For OCR texts: use regex for extraction!
"""

temp_aspects=["attribute","count","entity","relation"]
templates={temp_name:'\n'.join(open(os.path.join('../templates_llama',temp_name+'_temp.txt'),'r').readlines()) for temp_name in temp_aspects}
vis=0
attributes={}
print ('Number of predicted instances:',len(attributes))
vsg_pred=load_pkl(args.target_file)
print ('Number of captions:',len(vsg_pred)//8)
random.shuffle(names)
for name in names:
    #if vis>5:
    #    break
    idx=name_to_idx[name]
    if str(idx)+'_0' not in vsg_pred:
        continue
    vis+=1
    for i in range(8):
        
        if str(idx)+'_'+str(i) in attributes:
            continue
        if str(idx)+'_'+str(i) not in vsg_pred:
            continue
        pred=vsg_pred[str(idx)+'_'+str(i)]
        all_ext={}
        #print (pred)
        for temp_name in temp_aspects:
            examples=templates[temp_name]
            msg=examples + pred 
            message=[
                {"role":"system","content":"You are a helpful assistant!"},
                {"role":"user","content":msg}
            ]
            prompt = pipeline.tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True
                )
            attr=pipeline(prompt,
                          max_new_tokens=256,
                          pad_token_id = pipeline.tokenizer.eos_token_id,
                          eos_token_id=terminators)[0]["generated_text"][len(prompt):]
            all_ext[temp_name]=attr.split(':')[-1]
            #print ('\t',temp_name,attr)
        attributes[str(idx)+'_'+str(i)]=all_ext
        if vis%50==0:
            print (vis)
            print (pred)
            for temp_name in all_ext:
                print ('\t',temp_name,all_ext[temp_name])
    if vis%50==0:
        #print(pred,'\n\t',attr)
        pkl.dump(attributes,open(os.path.join('../decomp_results_llava_vsg_0.25','all_vsg_llama3.pkl'),'wb'))

#print(json.dumps(questions, indent=4))
pkl.dump(attributes,open(os.path.join('../decomp_results_llava_vsg_0.25','all_vsg_llama3.pkl'),'wb'))