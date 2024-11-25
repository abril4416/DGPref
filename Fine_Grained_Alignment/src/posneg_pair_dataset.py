import os
import json
import pickle as pkl
import numpy as np
import torch
import random
from PIL import Image
import random
import requests
import config
import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
import numpy as np
import transformers
from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX
from dpo_utils import get_diff_ids

from llava.constants import expand2square,OPENAI_CLIP_MEAN,process_anyres_image

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    # "%s <image>\nUSER: The text on the image is: %s. %s\nASSISTANT: %s"
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]#[INST+' ','\nUSER: '+TXT+'\nASST: '+RSP]
    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]
    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])
    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])
    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

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

#logp already available
class PosNeg_Pair_Data():
    #mem, off, harm
    #using opt for arguements, rather than args in other files
    def __init__(self,opt,tokenizer,mode='train',dataset="vsg",punish=False):
        super(PosNeg_Pair_Data,self).__init__()
        self.opt=opt
        #a list would be more optimal
        self.dataset=dataset
        #self.prompt_temp="<image>\nUSER: %s\nASSISTANT: %s"
        if opt.MODEL_NAME in ["llava-hf/llava-1.5-7b-hf","llava-hf/llava-1.5-13b-hf"]:
            self.prompt_temp="<image>\nUSER: %s\nASSISTANT: %s"
        elif opt.MODEL_NAME=='BAAI/Bunny-v1_0-3B':
            self.prompt_temp = "USER: <image>\n%s ASSISTANT: %s"
        elif opt.MODEL_NAME=='Qwen/Qwen-VL-Chat':
            """
            Creating data points similar to: https://github.com/cognitedata/Qwen-VL-finetune
                value in the data point
            """
            self.prompt_temp_user="Picture 1: <img>%s</img>\n%s"#image path and instruction
            self.prompt_temp_assist="%s"#response
            
        self.mode=mode
        self.punish=punish
        self.model_name=self.opt.MODEL_NAME.split('/')[-1]
        
        if self.model_name=='Qwen-VL-Chat':
            self.roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}
            system_message="You are a helpful assistant."
            self.im_start = tokenizer.im_start_id
            self.im_end = tokenizer.im_end_id
            self.nl_tokens = tokenizer('\n').input_ids
            self._system = tokenizer('system').input_ids + self.nl_tokens
            self.system_tokens = [self.im_start] + self._system + tokenizer(system_message).input_ids + [self.im_end] + self.nl_tokens
            self._user = tokenizer('user').input_ids + self.nl_tokens
            self._assistant = tokenizer('assistant').input_ids + self.nl_tokens

        self.tokenizer=tokenizer
        self.sep="\n"
        self.agent="ASSISTANT"
        self.user="USER"

        if self.opt.USE_BOTH and mode=='train':
            self.entities=self.load_train_data("vsg")
            self.entities.extend(self.load_train_data("coco"))
        else:
            self.entities=self.load_train_data(self.opt.DATA_NAME)
        if mode=='train' and opt.HUMAN_SIZE>0:
            self.entities=self.entities[:opt.HUMAN_SIZE]
        print (self.dataset,'\n\tLenght of data:',len(self.entities))
        #if self.opt.DEBUG:
        #    self.entities=self.entities[:16]

    def generate_input_target(self, ques, resp):
        full_prompt=(self.prompt_temp % (ques,resp))
        """
        Currently only support one round of conversation!
        Mask instructions as well as padding tokens
            The code defaultly focuses only on the respones part (i.e., not training on inputs)
        Currently no EOS token added
        """
        cur_len = 1
        if self.model_name in ["llava-1.5-7b-hf","llava-1.5-13b-hf"]:
            sep=self.sep+self.agent+': '#\nASSISTANT: ;
            input_ids = torch.stack([tokenizer_image_token(full_prompt, self.tokenizer, return_tensors='pt')], dim=0)
        elif self.model_name in ['Bunny-v1_0-3B']:
            sep=self.agent+': '
            input_ids = torch.stack([tokenizer_image_token(full_prompt, self.tokenizer, 
                                                           image_token_index=-200,  
                                                           return_tensors='pt')], dim=0)
        targets = input_ids.clone()
        total_len = int(targets.ne(self.tokenizer.pad_token_id).sum())
        targets[0,:cur_len] = IGNORE_INDEX
        parts = full_prompt.split(sep)
        parts[0] += sep #"[INST] \n"
        #round_len = len(tokenizer_image_token(full_prompt, self.tokenizer))
        instruction_len = len(tokenizer_image_token(parts[0], self.tokenizer)) - 2 #BOS and EOS
        targets[0,cur_len : cur_len + instruction_len] = IGNORE_INDEX
        """
        print ('Prompt:\n\t',full_prompt)
        print (input_ids)
        print (targets)
        """
        return input_ids, targets
    
    def generate_input_target_qwen(self, temp, cap, img_path):
        full_prompt={
            "user": (self.prompt_temp_user % (img_path, temp)),
            "assistant": (self.prompt_temp_assist % cap)
        }
        input_ids=[]
        targets=[]
        
        input_ids+=self.system_tokens
        targets+=[self.im_start]+[IGNORE_INDEX]*(len(self.system_tokens)-3)+[self.im_end]+self.nl_tokens
        
        role=self.roles['user']
        user_msg=full_prompt['user']
        input_id=self.tokenizer(role).input_ids+self.nl_tokens+self.tokenizer(user_msg).input_ids+[self.im_end]+self.nl_tokens
        input_ids+=input_id
        target = [self.im_start] + [IGNORE_INDEX] * (len(input_id)-3) + [self.im_end] + self.nl_tokens
        targets+=target
        
        role=self.roles['assistant']
        user_msg=full_prompt['assistant']
        input_id=self.tokenizer(role).input_ids+self.nl_tokens+self.tokenizer(user_msg).input_ids+[self.im_end]+self.nl_tokens
        input_ids+=input_id
        target = [self.im_start] + [IGNORE_INDEX] * len(self.tokenizer(role).input_ids)+\
        input_id[len(self.tokenizer(role).input_ids)+1:-2] + [self.im_end] + self.nl_tokens
        targets+=target
        assert len(input_ids) == len(targets)
        input_ids=torch.LongTensor(input_ids)
        targets=torch.LongTensor(targets)
        return input_ids, targets
    
    def load_train_data(self,data_name):
        entities=[]
        considered_aspects=self.opt.CONSIDERED_ASPECTS.split(',')
        model_name=self.opt.MODEL_NAME.split('/')[-1]
        """
        Currently, for GT, using chair 
            as the aspect
            rather than setting GT_OBJ
        """
        if self.punish:
            all_sampled_pairs=load_pkl(os.path.join("../pos_neg_pair_gen",'pos_neg_pairs-filter.pkl'))
        elif self.opt.GT_OBJ==False:
            all_sampled_pairs=load_pkl(
                os.path.join("../pos_neg_pair_gen",
                             '_'.join([data_name,model_name]),
                             'pos_neg_pairs-'+'_'.join(considered_aspects)+".pkl")
            )
            
        all_logp_file=load_pkl("../pos_neg_pair_gen/all_logp_"+'_'.join(
            [data_name,model_name]
        )+".pkl")
        print ('Length of all logp:',len(all_logp_file))
        names=list(all_sampled_pairs.keys())

        start_idx=0
        end_idx=len(names)
        if self.opt.DEBUG:
            start_idx=0
            end_idx=57
        if self.mode=='val':
            if self.opt.DEBUG==False:
                """
                Be careful!!! 
                This (placeholder) is selected according to the length of pos/neg file
                """
                end_idx=100 #select 300 examples from the training example as psuedo-val split
            else:
                end_idx=10
            place_holder_idx=random.randint(4500,4750)#make sure not out of index (12,600 currently ==> len names)
            start_idx+=place_holder_idx
            end_idx+=place_holder_idx
        print (self.mode, 'Starting: ',start_idx, ' Ending at: ',end_idx)
        template=open('../templates/objhal_temp.txt','r').readlines()
        print ('Loading pos and neg samples:', len(all_sampled_pairs))
        vis=0

        if data_name=='vsg':
            train_graphs=json.load(
                open(os.path.join(self.opt.VQA_PATH,'GQA/train_sceneGraphs.json') ,'r'))
            print ('Length of train graphs',len(train_graphs))
            graph_names=list(train_graphs.keys())
            name_to_idx={name:i for i,name in enumerate(graph_names)}
            IMG_DIR_1=os.path.join(self.opt.VQA_PATH,'VisualGenome/VG_100K')
            IMG_DIR_2=os.path.join(self.opt.VQA_PATH,'VisualGenome/VG_100K_2')
        elif data_name=='coco':
            img_dir=os.path.join(self.opt.VQA_PATH,'VQA/train2014')
            names=os.listdir(img_dir)
            if self.opt.DEBUG==False and self.mode=='train':
                end_idx=len(names)
       
        for i,name in enumerate(names):
            if i<start_idx:
                continue
            if i>=end_idx:
                break
            """
            For VSG: name is graph key
            For COCO: name is picture name
            """
            if name not in all_sampled_pairs:
                continue
            sampled_info=all_sampled_pairs[name]
            sampled_pairs=sampled_info['sample_pair']
            if len(sampled_pairs)==0:
                continue
            if data_name=='vsg':
                idx=name_to_idx[name]
                img_id=str(name)+'.jpg'
                if os.path.exists(os.path.join(IMG_DIR_1,img_id)):
                    img_path=os.path.join(IMG_DIR_1,img_id)
                elif os.path.exists(os.path.join(IMG_DIR_2,img_id)):
                    img_path=os.path.join(IMG_DIR_2,img_id)
            elif data_name=='coco':
                idx=i
                img_path=os.path.join(img_dir,name)
            
            for k,pair in enumerate(sampled_pairs):
                if self.opt.CONSIDERED_ASPECTS=='chair' and vis>=68414:#make it the same as previous baselines
                    print ('Cutting for chair split, to avoid OOM')
                    return entities
                prefer_cap=pair["prefer"]
                prefer_temp=template[pair["prefer_idx"]]
                reject_cap=pair["reject"]
                reject_temp=template[pair["reject_idx"]]
                """
                some heuristic approaches to avoid OOM problem
                    too long sentence ==> average length if removing len>100 ==> 88
                """
                prefer_len=len(prefer_cap.split(' '))
                reject_len=len(reject_cap.split(' '))
                if prefer_len>self.opt.MAX_LEN:
                    prefer_cap=' '.join(prefer_cap.split(' ')[:self.opt.MAX_LEN])
                if reject_len>self.opt.MAX_LEN:
                    reject_cap=' '.join(reject_cap.split(' ')[:self.opt.MAX_LEN])
                if self.model_name=='Qwen-VL-Chat':
                    prefer_input, prefer_label= self.generate_input_target_qwen(prefer_temp, prefer_cap, img_path)
                    reject_input, reject_label= self.generate_input_target_qwen(reject_temp, reject_cap, img_path)
                else:
                    prefer_input, prefer_label= self.generate_input_target(prefer_temp, prefer_cap)
                    reject_input, reject_label= self.generate_input_target(reject_temp, reject_cap)
                data_point={
                    "ref_win_per_token_logp":all_logp_file[str(idx)+'_'+str(pair["prefer_idx"])]['per_token_logp'],
                    "ref_rej_per_token_logp":all_logp_file[str(idx)+'_'+str(pair["reject_idx"])]['per_token_logp'],
                    "ref_win_avg_logp":all_logp_file[str(idx)+'_'+str(pair["prefer_idx"])]['avg_logp'],
                    "ref_rej_avg_logp":all_logp_file[str(idx)+'_'+str(pair["reject_idx"])]['avg_logp'],
                    "ref_win_logp":all_logp_file[str(idx)+'_'+str(pair["prefer_idx"])]['logp'],
                    "ref_rej_logp":all_logp_file[str(idx)+'_'+str(pair["reject_idx"])]['logp'],
                    "win_input_ids":prefer_input.squeeze(0),
                    "rej_input_ids":reject_input.squeeze(0),
                    "win_labels":prefer_label.squeeze(0),
                    "rej_labels":reject_label.squeeze(0),
                    "img_path":img_path,
                    "idx":idx,
                    "name":name
                }
                #print (data_point)
                entities.append(data_point)
                vis+=1
        if self.opt.HALF_VSG:
            entities=entities[:len(entities)//2]
        return entities

    def __len__(self):
        return len(self.entities)
        
    def __getitem__(self, i):
        data_point=self.entities[i]
        name=data_point['name']
        idx=data_point['idx']
        img_path=data_point['img_path']
        """
        For better CPU utilization
            load images during training
        """
        batch={
            #'pixel_values':pixel_values,
            'name':name,
            'idx':idx,
            'img_path': img_path,
            'win_input_ids':data_point['win_input_ids'],
            'win_labels':data_point['win_labels'],
            'rej_input_ids':data_point['rej_input_ids'],
            'rej_labels':data_point['rej_labels'],
            "ref_win_per_token_logp":data_point['ref_win_per_token_logp'],
            "ref_rej_per_token_logp":data_point['ref_rej_per_token_logp'],
            "ref_win_avg_logp":data_point['ref_win_avg_logp'],
            "ref_rej_avg_logp":data_point['ref_rej_avg_logp'],
            "ref_win_logp":data_point['ref_win_logp'],
            "ref_rej_logp":data_point['ref_rej_logp'],
        }
        return batch
    
#the per_token_logp also need padding to unified length!!
#https://github.com/RLHF-V/RLAIF-V/blob/main/muffin/train/train_muffin.py#L38
class DataCollatorDPODataset(object):
    """Code from LLaVA Project"""

    def __init__(self,model_name,tokenizer,image_processor,
                 bunny_config, model_dtype):
        self.tokenizer=tokenizer
        self.model_name=model_name
        self.img_processor=image_processor
        self.bunny_config=bunny_config
        self.model_dtype=model_dtype
        self.mod_token_weight=1.0

    def formulate_tensor(self, input_ids, labels):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        return input_ids, labels

    def __call__(self, instances):
        #print (instances)
        win_input_ids, win_labels = tuple([instance[key] for instance in instances]
                                  for key in ("win_input_ids", "win_labels"))
        rej_input_ids, rej_labels = tuple([instance[key] for instance in instances]
                                  for key in ("rej_input_ids", "rej_labels"))
        win_input_ids, win_labels = self.formulate_tensor(win_input_ids, win_labels)
        rej_input_ids, rej_labels = self.formulate_tensor(rej_input_ids, rej_labels)
        #attention mask ignore here ==> concatenation will ruin the current attention mask
        #pad when loading each batch
        #loading the logp information for reference
        ref_win_logp = torch.as_tensor(
            [x['ref_win_logp'] for x in instances])
        ref_rej_logp = torch.as_tensor(
            [x['ref_rej_logp'] for x in instances])
        ref_win_avg_logp = torch.as_tensor([x['ref_win_avg_logp'] for x in instances])
        ref_rej_avg_logp = torch.as_tensor([x['ref_rej_avg_logp'] for x in instances])
        ref_win_per_token_logp = [torch.as_tensor(
            x['ref_win_per_token_logp']) for x in instances]
        ref_rej_per_token_logp = [torch.as_tensor(
            x['ref_rej_per_token_logp']) for x in instances]

        ref_win_per_token_logp = torch.nn.utils.rnn.pad_sequence(
            ref_win_per_token_logp, batch_first=True, padding_value=0)
        ref_rej_per_token_logp = torch.nn.utils.rnn.pad_sequence(
            ref_rej_per_token_logp, batch_first=True, padding_value=0)
        
        #generating token weights
        win_token_weight = torch.ones_like(ref_win_per_token_logp)
        rej_token_weight = torch.ones_like(ref_rej_per_token_logp)

        for idx, (w, r, wl, rl, wlogp, rlogp) in enumerate(zip(win_input_ids, rej_input_ids, win_labels, rej_labels, ref_win_per_token_logp, ref_rej_per_token_logp)):
            valid_w = w[1:]
            valid_r = r[1:]
            min_match_size = 3
            r_mod, w_mod = get_diff_ids(
                valid_r.tolist(), valid_w.tolist(), min_match_size=min_match_size)
            r_mod_tokens = valid_r[r_mod]
            w_mod_tokens = valid_w[w_mod]
            win_token_weight[idx][w_mod] = self.mod_token_weight
            rej_token_weight[idx][r_mod] = self.mod_token_weight

        
        batch = {
            "win_dict":{
                "input_ids":win_input_ids,
                "labels":win_labels,
                "logp":ref_win_logp,
                "avg_logp":ref_win_avg_logp,
                "per_token_logp":ref_win_per_token_logp[:,:win_input_ids.size(1)-1],
                "token_weight":win_token_weight
                },
            "rej_dict":{
                "input_ids":rej_input_ids,
                "labels":rej_labels,
                "logp":ref_rej_logp,
                "avg_logp":ref_rej_avg_logp,
                "per_token_logp":ref_rej_per_token_logp[:,:rej_input_ids.size(1)-1],
                "token_weight":rej_token_weight
            }
        }

        if self.model_name in ["llava-1.5-7b-hf","llava-1.5-13b-hf"]:
            #im = expand2square(im, tuple(int(x*255) for x in OPENAI_CLIP_MEAN))
            pixel_values=self.img_processor([
                expand2square(Image.open(instance['img_path']).convert('RGB') , tuple(int(x*255) for x in OPENAI_CLIP_MEAN))
                #process_anyres_image(Image.open(instance['img_path']).convert('RGB'))
                for instance in instances], 
                return_tensors='pt').pixel_values
            batch['pixel_values']=pixel_values
        elif self.model_name=='Bunny-v1_0-3B':
            pixel_values=[self.img_processor([Image.open(instance['img_path']).convert('RGB')], 
                                            self.bunny_config).to(dtype=self.model_dtype)  
                          for instance in instances
                         ]
            pixel_values=torch.cat(pixel_values,dim=0)
            batch['images']=pixel_values
        elif self.model_name=='Qwen-VL-Chat':
            batch['images']=None
            #for instance in instances:
            #    print (Image.open(instance['img_path']).convert('RGB').size)
        # print(input_ids.shape, input_ids.device, batch['images'].shape, batch['images'].device)
        if "name" in instances[0]:
            batch['name']=[instance['name'] for instance in instances]
        if "idx" in instances[0]:
            batch['idx']=[instance['idx'] for instance in instances]
        if 'img_path' in instances[0]:
            batch['img_path']=[instance['img_path'] for instance in instances]
        return batch