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
import torch 

from constant import IMAGE_CONSTANT

#for experts
from transformers import AutoProcessor, OwlViTForObjectDetection
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import pipeline
import easyocr

def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data

def load_json(path):
    data=json.load(open(path,'r'))
    return data

def formulate_tensor(input_ids, pad_token_id,max_length):
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=pad_token_id)
    input_ids = input_ids[:, :max_length]
    return input_ids

"""
this is to avoid OOM
    too long object list or too many vqa questions per instance
    chunk it to several small lists
"""
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
"""
Define heuristics in modules
Define arguments fed into the modules
    Receiving processed programs
Tried Grounding DINO for attribute recognition DETECT(attr, obj)
    the results are not satisfactory
When formulating a question, employ a grammar/fluency evaluator 
    avoid some errors in decomposition
Also need to calculate the percentage of heuristic spatial relations
    justify why we define heuristics ==> hard for models but easy with heuristics
    adopt code here: https://github.com/RAIVNLab/sugar-crepe/blob/main/text_model_eval.py
"""

class ExpertModules():
    def __init__(self, args, debug_mode=False):
        self.args=args
        self.debug_mode=debug_mode

        #constant variables
        self.question_template={
            2: "Question: Is the %s %s? Short answer:",
            3: "Question: Is the %s %s %s? Short answer:"
        }

        self.grammar_expert= pipeline("text-classification", model=args.GRAMMAR_MODEL_NAME, device=0)
        self.obj_expert={
            "processor":AutoProcessor.from_pretrained(args.OBJDET_MODEL_NAME),
            "model":OwlViTForObjectDetection.from_pretrained(args.OBJDET_MODEL_NAME).cuda()
        }
        self.obj_expert["model"].eval()
        self.text_expert= easyocr.Reader(['en'])
        self.vqa_expert={
            "processor":Blip2Processor.from_pretrained(args.VQA_MODEL_NAME),
            "model":Blip2ForConditionalGeneration.from_pretrained(args.VQA_MODEL_NAME,  device_map="auto")
        }
        #self.vqa_expert["processor"].tokenizer.model_max_length=self.args.MAX_QUES_LENGTH
        self.vqa_expert["model"].eval()

    def obj_module(self, image, raw_obj_check_list):
        chunk_lists=list(chunks(raw_obj_check_list, self.args.CHUNK_LENGTH))
        #this is for saving det info... Pls not be confused by the name
        check_info={o:{'scores':[],'bboxs':[]} for o in raw_obj_check_list}
        score_info={o:-1 for o in raw_obj_check_list}
        for text in chunk_lists:
            inputs = self.obj_expert["processor"](text=[text], images=image, padding=True, truncation=True, return_tensors="pt")
            #print (text)
            #print (inputs)
            with torch.no_grad():
                outputs = self.obj_expert["model"](**inputs.to(self.obj_expert["model"].device))
            outputs['pred_boxes']=outputs['pred_boxes'].cpu()
            outputs['logits']=outputs['logits'].cpu()
            target_sizes = torch.Tensor([image.size[::-1]])
            results =self.obj_expert["processor"].post_process_object_detection(
                outputs=outputs, threshold=0.25, target_sizes=target_sizes
                )
            boxes, scores, labels = results[0]["boxes"], results[0]["scores"].cpu(), results[0]["labels"]
            for box, score, label in zip(boxes, scores, labels):
                box = [round(i, 2) for i in box.tolist()]
                if len(box):
                    score_info[text[label]]=1
                else:
                    score_info[text[label]]=-1
                check_info[text[label]]['scores'].append(score.item())
                check_info[text[label]]['bboxs'].append(box)
        return check_info, score_info

    def count_module(self, check_items, det_info):
        """
        Check if objects exist firstly ==> if not, assigning a special value -2
        """
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
            if self.debug_mode:
                print (obj, len(det_info[obj]['scores']))
        return check_info

    def scale_module(self, image, check_items, det_info):
        """
        Check if objects exist firstly ==> if not, assigning a special value -2
        Define heuristics for scale checking
        Heuristics for scale arguments:
        large, small, long, short, tall
        """
        check_info={}
        width, height= image.size
        for item in check_items:
            info=item.split(self.args.SPLIT_TERM)
            obj=info[1]
            if len(det_info[obj]['scores'])==0:
                check_info[item]=-2
                continue
            scale=info[0]
            scores=det_info[obj]["scores"]
            bboxs=det_info[obj]["bboxs"]
            flag=False
            for k,score in enumerate(scores):
                if flag:
                    break
                if score<self.args.BBOX_CONFIDENCE:
                    continue
                (xmin, ymin, xmax, ymax)=bboxs[k]
                obj_width=xmax-xmin
                obj_height=ymax-ymin
                if self.debug_mode:
                    print ('Scale check:',item)
                    print ('\t',obj_width/width, obj_height/height)
                if scale=='large':
                    if (obj_width/width)>self.args.LARGE_WIDTH or (obj_height/height)>self.args.LARGE_HEIGHT:
                        flag=True
                elif scale=='small':
                    if (obj_width/width)<self.args.SMALL_WIDTH and (obj_height/height)<self.args.SMALL_HEIGHT:
                        flag=True
                elif scale=='long':
                    if (obj_width/width)>self.args.LONG_SIZE or (obj_height/height)>self.args.LONG_SIZE:
                        flag=True
                elif scale=='short':
                    if (obj_width/width)<self.args.SHORT_SIZE and (obj_height/height)<self.args.SHORT_SIZE:
                        flag=True
                elif scale=='tall':
                    if (obj_height/height)>self.args.HEIGHT_SIZE:
                        flag=True
            if flag:
                check_info[item]=1
            else:
                check_info[item]=-1
        return check_info
    
    def spatial_module(self, image, check_items, det_info):
        """
        Check if objects exist firstly ==> if not, assigning a special value -2
        Define heuristics for spatial check
        Heuristics for spatial arguments:
            left, right, top, bottom, near
            see program_to_tool.py get_mapper
        """
        check_info={}
        width, height= image.size
        for item in check_items:
            info=item.split(self.args.SPLIT_TERM)
            sub=info[0]
            obj=info[-1]
            if (len(det_info[sub]['scores'])==0 or (obj not in IMAGE_CONSTANT and len(det_info[obj]['scores'])==0)):
                check_info[item]=-2
                continue
            rela=info[1]
            sub_scores=det_info[sub]["scores"]
            sub_bboxs=det_info[sub]["bboxs"]
            if obj not in IMAGE_CONSTANT:
                obj_scores=det_info[obj]["scores"]
                obj_bboxs=det_info[obj]["bboxs"]
            else:
                obj_scores=[1.0]
                obj_bboxs=[[0.0,0.0,width, height]]
            flag=False
            """
            Here, heuristics are defined with the central coordinate
            """
            for k,sub_score in enumerate(sub_scores):
                if flag:
                    break
                if sub_score<self.args.BBOX_CONFIDENCE:
                    continue
                (sxmin, symin, sxmax, symax)=sub_bboxs[k]
                sub_width=sxmax-sxmin
                sub_height=symax-symin
                sub_center_x=(sxmax+sxmin)/2.0
                sub_center_y=(symax+symin)/2.0
                for j, obj_score in enumerate(obj_scores):
                    if obj_score<self.args.BBOX_CONFIDENCE:
                        continue
                    (oxmin, oymin, oxmax, oymax)=obj_bboxs[j]
                    obj_width=oxmax-oxmin
                    obj_height=oymax-oymin
                    obj_center_x=(oxmax+oxmin)/2.0
                    obj_center_y=(oymax+oymin)/2.0
                    if self.debug_mode:
                        print ('Spatial check: ',item)
                        print ('\t',sub_center_x,obj_center_x,sub_center_x/width,obj_center_x/width)
                        print ('\t',sub_center_y,obj_center_y,sub_center_y/height,obj_center_y/height)
                    if rela=='left':
                        if sub_center_x<obj_center_x:
                            flag=True
                    elif rela=='right':
                        if sub_center_x>obj_center_x:
                            flag=True
                    elif rela=='top':
                        if sub_center_y>obj_center_y:
                            flag=True
                    elif rela=='bottom':
                        if sub_center_y<obj_center_y:
                            flag==True
                    elif rela=='near':
                        if abs(sub_center_x/width-obj_center_x/width)<self.args.DIFF_THRESHOLD or abs(sub_center_y/height-obj_center_y/height)<self.args.DIFF_THRESHOLD:
                            flag=True
            if flag:
                check_info[item]=1
            else:
                check_info[item]=-1
        return check_info

    def text_module(self, image_path, check_texts, details=False):
        """
        details: whether return coordinates of detected texts
        """
        check_info={"details":[],"scores":{},"det_texts":[]}
        det_ocr_texts=self.text_expert.readtext(image_path, detail =int(details))
        if details:
            check_info["details"]=det_ocr_texts
            process_det_ocr=[]
            for info in det_ocr_texts:
                text=info[1].lower().strip().replace(".", "").replace(",", "").replace("?", "").replace("!", "").replace("*", "")
                process_det_ocr.append(text)
                check_info["det_texts"].append(text)
        else:
            process_det_ocr=[
                text.lower().strip().replace(".", "").replace(",", "").replace("?", "").replace("!", "").replace("*", "") for text in det_ocr_texts
            ]
            check_info["det_texts"]=process_det_ocr
        for text in check_texts:
            if text.lower().strip().replace(".", "").replace(",", "").replace("?", "").replace("!", "").replace("*", "") in process_det_ocr:
                check_info["scores"][text]=1
            else:
                check_info["scores"][text]=-1
        return check_info
            
    def validate_ques_module(self, question):
        with torch.no_grad():
            output = self.grammar_expert(question)[0]
            valid_score = output['score'] if output['label'] == 'LABEL_1' else 1 - output['score']
        return valid_score

    def arguments_to_question(self, check_item):
        info=check_item.split(self.args.SPLIT_TERM)
        ques_temp=self.question_template[len(info)]
        if len(info)==2:
            question=(ques_temp % (info[1], info[0]))
        elif len(info)==3:
            question=(ques_temp % (info[0], info[1], info[-1]))
        else:
            question=""
        return question
    
    def vqa_module(self, image, check_items, check_type, det_info, validate_ques=False):
        """
        Map arguments into questions ==> call arguments_to_question
        Call validate_question_module, optionally
            checking whether the question is plausible/grammatically correct
            if the question is not valid, give the score, zero
        """
        questions={}
        check_info={"questions":{}, "scores":{}}
        if validate_ques:
            check_info["question_scores"]={}
        for item in check_items:
            #check objects
            info=item.split(self.args.SPLIT_TERM)
            if check_type=='attribute_check':
                obj=info[-1]
                if (len(det_info[obj]['scores'])==0):
                    check_info['scores'][item]=-2
                    question=""
                    questions[item]=""
                else:
                    question=self.arguments_to_question(item)
                    questions[item]=question
            elif check_type=='relation_check':
                sub=info[0]
                obj=info[-1]
                if (len(det_info[sub]['scores'])==0 or (obj not in IMAGE_CONSTANT and len(det_info[obj]['scores'])==0)):
                    check_info['scores'][item]=-2
                    questions[item]=""
                    question=""
                else:
                    question=self.arguments_to_question(item)
                    questions[item]=question
            if validate_ques and len(question)>0:
                valid_score=self.validate_ques_module(question)
                check_info["question_scores"][question]=valid_score
                if self.debug_mode:
                    print (question)
                    print ('\tValidation scores:',valid_score)
                if valid_score<self.args.VALID_QUES_SCORE:
                    check_info['scores'][item]=0
        """
        this filter out all questions: 
            1) objects not exist; 2) low valid scores
        """
        check_info["questions"]=questions
        valid_question_mapper={questions[item]:item for item in check_items if (item not in check_info['scores'])}
        valid_question_list=list(valid_question_mapper.keys())
        question_chunks=chunks(valid_question_list, self.args.CHUNK_LENGTH_QUES)
        for ques_chunk in question_chunks:
            input_images=[image]*len(ques_chunk)
            """
            pixel_values=self.vqa_expert["processor"].image_processor(input_images, return_tensors="pt").to(self.vqa_expert["model"].device)
            input_ids=[self.vqa_expert["processor"].tokenizer(ques, return_tensors="pt").input_ids.squeeze() for ques in ques_chunk]
            print (input_ids)
            input_ids=formulate_tensor(input_ids,
                                       self.vqa_expert["processor"].tokenizer.pad_token_id,
                                       self.args.MAX_QUES_LENGTH).to(self.vqa_expert["model"].device)
            attention_mask=input_ids.ne(self.vqa_expert["processor"].tokenizer.pad_token_id).to(self.vqa_expert["model"].device)
            """
            inputs=self.vqa_expert["processor"](input_images, ques_chunk,  
                                                padding="max_length", truncation=True, max_length=self.args.MAX_QUES_LENGTH,
                                                return_tensors="pt").to(self.vqa_expert["model"].device)
            
            with torch.no_grad():
                out = self.vqa_expert["model"].generate(
                    #pixel_values=pixel_values,
                    #input_ids=input_ids,
                    #attention_mask=attention_mask,
                    **inputs,
                    length_penalty=-1,do_sample=True,num_beams=2)
            answers=self.vqa_expert["processor"].batch_decode(out, skip_special_tokens=True)
            for i,ques in enumerate(ques_chunk):
                ans=answers[i].lower()
                item=valid_question_mapper[ques]
                if self.debug_mode:
                    print (ques)
                    print ('\tVQA answer:',ans)
                if 'yes' in ans:
                    check_info['scores'][item]=1
                else:
                    check_info['scores'][item]=-1
        return check_info
