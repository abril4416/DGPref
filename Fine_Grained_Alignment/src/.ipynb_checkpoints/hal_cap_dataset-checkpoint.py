import os
import json
import pickle as pkl
import numpy as np
import torch
import random
from PIL import Image
import random
import requests

def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data
    
def load_json(path):
    data=json.load(open(path,'rb'))
    return data

def load_jsonl(path):
    total_info=[]
    with open(path,'rb')as f:
        d=f.readlines()
    for i,info in enumerate(d):
        data=json.loads(info)
        total_info.append(data)
    return total_info

class Hal_Cap_Data():
    #using opt for arguements, rather than args in other files
    def __init__(self,opt,dataset,mode='val'):
        super(Hal_Cap_Data,self).__init__()
        self.opt=opt
        self.dataset=dataset
        if opt.MODEL_NAME in ["llava-hf/llava-1.5-7b-hf","llava-hf/llava-1.5-13b-hf"]:
            self.prompt_temp="<image>\nUSER: %s\nASSISTANT:"
            if dataset in ['gqa','text-vqa','vizwiz']:
                system="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
                self.prompt_temp=system+" USER: <image>\n%s ASSISTANT:"
            #self.prompt_temp="USER: <image>\n%s ASSISTANT:"
        elif opt.MODEL_NAME=='BAAI/Bunny-v1_0-3B':
            header="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "
            self.prompt_temp = "USER: <image>\n%s ASSISTANT:"
        elif opt.MODEL_NAME=="Qwen/Qwen-VL-Chat":
            self.prompt_temp="%s"
        """
        train mode is used to generate DPO data
            per image multiple responses (sampling or different templates)
        """
        if mode=='val':
            self.entities=self.load_data()
        else:
            self.entities=self.load_repeat_data()
        print (self.dataset,'\n\tLenght of data:',len(self.entities))
        if self.opt.DEBUG:
            self.entities=self.entities[:16]

    """
    Something wrong with the load_repeat_data
        be careful when generating for train --  for other datasets
        but may not be used for other datasets -- currently only vsg needs the implementation
    This implementation is to generate multiple responses per image
        Sample to rank responses ==> preferred vs. rejected (currently using eight different templates for detailed image description generation)
    """
    def load_repeat_data(self):
        entities=[]
        if self.dataset=='vsg':
            """
            Be careful here, loading training data 
            """
            val_all=load_json(os.path.join(self.opt.VQA_PATH,'GQA/train_sceneGraphs.json'))
            data=[]
            IMG_DIR_1=os.path.join(self.opt.VQA_PATH,'VisualGenome/VG_100K')
            IMG_DIR_2=os.path.join(self.opt.VQA_PATH,'VisualGenome/VG_100K_2')
            """
            This is just to starts with VG in VG scene graphs
                (For ablations, because VG scene graphs have fine-grained annotations, e.g., objects, attributes and relations)
                so we need this mapper
            """
            for name in val_all.keys():
                cur_row=val_all[name]
                cur_row['image_id']=name
                data.append(cur_row)
        elif self.dataset=='coco':
            """
            Be careful here, loading training data 
            Use no human annotations 
            """
            img_dir=os.path.join(self.opt.VQA_PATH,'VQA/train2014')
            data=os.listdir(img_dir)
        templates=open('../templates/objhal_temp.txt','r').readlines()
        for i,row in enumerate(data):
            """
            Be careful, no shuffle here, or the orders will be messed up
                if shuffling when checking results, remember to use a mapper
            Load inputs during generation -- better debugging
                will have OOM (CPU) if loading all images (Image.open()) in dataloader
            """
            for k,text in enumerate(templates):
                prompt=(self.prompt_temp % text)
                if self.dataset=='vsg':
                    img_id=row['image_id']+'.jpg'
                    if os.path.exists(os.path.join(IMG_DIR_1,img_id)):
                        img_path=os.path.join(IMG_DIR_1,img_id)
                    elif os.path.exists(os.path.join(IMG_DIR_2,img_id)):
                        img_path=os.path.join(IMG_DIR_2,img_id)
                elif self.dataset=='coco':
                    img_id=row
                    img_path=os.path.join(img_dir,img_id)
                entry={
                    'question':text,
                    'prompt':prompt,
                    'img_path':img_path,
                    'img_id':img_id,
                    'idx':str(i)+'_'+str(k) # i denots the i-th location in trainGraphs, without shuffling; k denotes the k-th template
                }
                entities.append(entry)
        return entities
        
    def load_data(self):
        entities=[]
        if self.dataset=='objhal':
            data=load_jsonl(os.path.join(self.opt.VQA_PATH,'Hal_Benchs/obj_halbench_300_with_image.jsonl'))
        elif self.dataset=='mmhal':
            data=load_json(os.path.join(self.opt.VQA_PATH,'Hal_Benchs/mmhal-bench_answer_template.json'))
        elif self.dataset=='vsg':
            val_all=load_json(os.path.join(self.opt.VQA_PATH,'GQA/val_sceneGraphs.json'))
            data=[]
            IMG_DIR_1=os.path.join(self.opt.VQA_PATH,'VisualGenome/VG_100K')
            IMG_DIR_2=os.path.join(self.opt.VQA_PATH,'VisualGenome/VG_100K_2')
            for name in val_all.keys():
                cur_row=val_all[name]
                cur_row['image_id']=name
                data.append(cur_row)
            templates=open('../templates/objhal_temp.txt','r').readlines()
        elif self.dataset=='amber':
            data=load_json(os.path.join(self.opt.VQA_PATH,'AMBER/data/query/query_generative.json'))
        elif self.dataset in ['gqa']:
            #12578
            gqa_data=load_json(os.path.join(self.opt.VQA_PATH,'GQA','testdev_balanced_questions.json'))
            data=[]
            for idx in gqa_data:
                gqa_row=gqa_data[idx]
                gqa_row['image_id']=idx#using the key as the unique indexing
                data.append(gqa_row)
            print ('Loading GQA dataset:',len(data))
        elif self.dataset=='vqa':
            data=load_json(os.path.join(self.opt.VQA_PATH,'VQA/val_all.json'))[:2000]
        elif self.dataset=='text-vqa':
            #3166
            data=load_jsonl(os.path.join(self.opt.VQA_PATH,'text-vqa/llava_textvqa_val_v051_ocr.jsonl'))
            print ('Loading Text-VQA dataset:',len(data))
        elif self.dataset=='vizwiz':
            #4319
            data=load_json(os.path.join(self.opt.VQA_PATH,'vizwiz/val.json'))
            print ('Loading Text-VQA dataset:',len(data))
        elif self.dataset=='llava-bench':
            data=load_jsonl(os.path.join(self.opt.VQA_PATH,'llava-bench-in-the-wild/questions.jsonl'))
            print ('Loading LLaVA Bench Wild dataset:',len(data))
        for i,row in enumerate(data):
            """
            For optimization, load inputs during generation -- better debugging
            """
            if self.dataset=='vsg':
                """
                randomly pick a template from eight templates
                    following previous works, to guarantee the robustness of models towards different templates
                """
                idx=random.randint(0,7)
                text=templates[idx]
            elif self.dataset=='amber':
                text=row['query']
            elif self.dataset in ['llava-bench','text-vqa']:
                text=row['text']
            else:
                text=row['question']
            """
            For text-vqa using LLaVA provided data
                already added this instruction
            """
            if self.dataset in ['gqa','vizwiz']:
                if self.dataset=='vizwiz':
                    text=text+'\nWhen the provided information is insufficient, respond with \'Unanswerable\'.'
                text=text+'\nAnswer the question using a single word or phrase.'
            prompt=(self.prompt_temp % text)
            if self.dataset in ['objhal','vqa']:
                if self.dataset=='objhal':
                    img_id=str(row['image_id'])
                elif self.dataset=='vqa':
                    img_id=str(row['img_id'])
                img_id='COCO_val2014_'+img_id.zfill(12)+'.jpg'
                img_path=os.path.join(self.opt.VQA_PATH,'VQA/val2014',img_id)
            elif self.dataset in ['mmhal']:
                img_id=row['image_id']+'.jpg'
                if img_id=='7b6eed2a50ffd046.jpg':
                    continue
                img_path=os.path.join(self.opt.VQA_PATH,'MMHal_Bench',img_id)
            elif self.dataset=='vsg':
                img_id=row['image_id']+'.jpg'
                if os.path.exists(os.path.join(IMG_DIR_1,img_id)):
                    img_path=os.path.join(IMG_DIR_1,img_id)
                elif os.path.exists(os.path.join(IMG_DIR_2,img_id)):
                    img_path=os.path.join(IMG_DIR_2,img_id)
            elif self.dataset=='amber':
                img_id=row['image']
                row['image_id']=row['id']
                img_path=os.path.join(self.opt.VQA_PATH,'AMBER/images',img_id)
            elif self.dataset=='gqa':
                img_id=row['imageId']
                img_path=os.path.join(self.opt.VQA_PATH,'GQA/images',img_id+'.jpg')
            elif self.dataset=='text-vqa':
                img_id=row['image']
                img_path=os.path.join(self.opt.VQA_PATH,'text-vqa/train_images',img_id)
                row['image_id']=i
            elif self.dataset=='vizwiz':
                img_id=row['image']
                img_path=os.path.join(self.opt.VQA_PATH,'vizwiz/val_images',img_id)
                row['image_id']=i
            elif self.dataset=='llava-bench':
                img_id=row['image']
                img_path=os.path.join(self.opt.VQA_PATH,'llava-bench-in-the-wild/images',img_id)
            if self.dataset in ['llava-bench']:
                uniq_idx=row['question_id']
            elif self.dataset in ['vqa']:
                uniq_idx=row['idx']
            else:
                """
                GQA: key 
                text-vqa: question_id
                vizwiz: indexing of the row
                """
                uniq_idx=str(row['image_id'])
            entry={
                'question':text,
                'prompt':prompt,
                'img_path':img_path,
                'raw_img_id':uniq_idx,
                'img_id':img_id,
                'idx':str(i)
            }
            entities.append(entry)
        return entities
        