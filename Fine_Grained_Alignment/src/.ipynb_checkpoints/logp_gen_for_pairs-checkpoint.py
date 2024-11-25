import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
from llava.constants import expand2square,OPENAI_CLIP_MEAN

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

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

def log_hyperpara(opt):
    dic = vars(opt)
    for k,v in dic.items():
        print(k + ' : ' + str(v))

#adopted from RLAIF-V: https://github.com/RLHF-V/RLAIF-V/blob/main/muffin/eval/muffin_inference_logp.py#L82
def get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, return_per_token_logp=False, return_all=False, tokenizer=None) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape, f'logits.shape[:-1]={logits.shape[:-1]}, labels.shape={labels.shape}'

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2,
                                   index=labels.unsqueeze(2)).squeeze(2)

    log_prob = (per_token_logps * loss_mask).sum(-1)
    average_log_prob = log_prob / loss_mask.sum(-1)

    if return_per_token_logp:
        return per_token_logps

    if return_all:
        return per_token_logps, log_prob, average_log_prob
    return log_prob, average_log_prob

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

class Pref_Pair_Data():
    #mem, off, harm
    #using opt for arguements, rather than args in other files
    def __init__(self,opt,dataset="vsg"):
        super(Pref_Pair_Data,self).__init__()
        self.opt=opt
        #a list would be more optimal
        self.dataset=dataset
        if opt.MODEL_NAME in ["llava-hf/llava-1.5-7b-hf","llava-hf/llava-1.5-13b-hf"]:
            self.prompt_temp="<image>\nUSER: %s\nASSISTANT: %s"
        elif opt.MODEL_NAME=='BAAI/Bunny-v1_0-3B':
            #header="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "
            self.prompt_temp = "USER: <image>\n%s ASSISTANT: %s"
        elif opt.MODEL_NAME=='Qwen/Qwen-VL-Chat':
            """
            Creating data points similar to: https://github.com/cognitedata/Qwen-VL-finetune
                value in the data point
            """
            self.prompt_temp_user="Picture 1: <img>%s</img>\n%s"#image path and instruction
            self.prompt_temp_assist="%s"#response
            
        self.sep="\n"
        self.agent="ASSISTANT"
        self.user="USER"

        self.entities=self.load_train_data()
        print (self.dataset,'\n\tLenght of data:',len(self.entities))
        #if self.opt.DEBUG:
        #    self.entities=self.entities[:16]

    def generate_input_with_temp(self, ques, resp):
        full_prompt=(self.prompt_temp % (ques,resp))
        return full_prompt
    
    def load_train_data(self):
        entities=[]
        considered_aspects=self.opt.CONSIDERED_ASPECTS.split(',')
        if self.opt.DATA_NAME=='vsg' and ('llava-hf' in self.opt.MODEL_NAME or "Qwen-VL-Chat" in self.opt.MODEL_NAME):
            print ('Reusing LLaVA generated captions!!!!')
            vsg_pred=load_pkl("../results/gen_results/raw_gen/llava-1.5-7b-hf/vsg.pkl")
        elif self.opt.DATA_NAME=='coco' and ('llava-hf' in self.opt.MODEL_NAME or "Qwen-VL-Chat" in self.opt.MODEL_NAME):
            print ('Reusing LLaVA generated captions!!!!')
            vsg_pred=load_pkl("../results/gen_results/raw_gen/llava-1.5-7b-hf/coco.pkl")
        else:
            vsg_pred=load_pkl(os.path.join(
                "../results/gen_results/raw_gen", self.opt.MODEL_NAME.split('/')[1], self.opt.DATA_NAME+'.pkl'))
        print ('Length of all caption data:',len(vsg_pred)//8)
        template=open('../templates/objhal_temp.txt','r').readlines()

        if self.opt.DATA_NAME=='vsg':
            train_graphs=json.load(
                open(os.path.join(self.opt.VQA_PATH,'GQA/train_sceneGraphs.json') ,'r'))
            print ('Length of train graphs',len(train_graphs))
            names=list(train_graphs.keys())
            name_to_idx={name:i for i,name in enumerate(names)}
        elif self.opt.DATA_NAME=='coco':
            img_dir=os.path.join(self.opt.VQA_PATH,'VQA/train2014')
            names=os.listdir(img_dir)
        if self.opt.DEBUG:
            names=names[:17]
        IMG_DIR_1='/PATH/VQA/VisualGenome/VG_100K'
        IMG_DIR_2='/PATH/VQA/VisualGenome/VG_100K_2'
        for k,name in enumerate(names):
            if self.opt.DATA_NAME=='vsg':
                idx=name_to_idx[name]
            else:
                idx=k
            if str(idx)+'_0' not in vsg_pred:
                continue
            if self.opt.DATA_NAME=='vsg':
                img_id=str(name)+'.jpg'
                if os.path.exists(os.path.join(IMG_DIR_1,img_id)):
                    img_path=os.path.join(IMG_DIR_1,img_id)
                elif os.path.exists(os.path.join(IMG_DIR_2,img_id)):
                    img_path=os.path.join(IMG_DIR_2,img_id)
            elif self.opt.DATA_NAME=='coco':
                img_path=os.path.join(img_dir,name)
            
            for i in range(8):
                if str(idx)+'_'+str(i) not in vsg_pred:
                    continue
                cap=vsg_pred[str(idx)+'_'+str(i)]
                temp=template[i]
                #print (temp,'\n\t',cap)
                #input, label= self.generate_input_target(temp, cap)
                if self.opt.MODEL_NAME=='Qwen/Qwen-VL-Chat':
                    full_prompt={
                        "user": (self.prompt_temp_user % (img_path, temp)),
                        "assistant": (self.prompt_temp_assist % cap)
                    }
                else:
                    full_prompt= self.generate_input_with_temp(temp, cap)
                """
                Optimze to reduce the burden of CPU
                """
                data_point={
                    "full_prompt":full_prompt,
                    "img_path":img_path,
                    "idx":str(idx)+'_'+str(i),
                    "name":name
                }
                entities.append(data_point)
        return entities

    def __len__(self):
        return len(self.entities)
        
    def __getitem__(self, i):
        data_point=self.entities[i]
        name=data_point['name']
        idx=data_point['idx']
        img_path=data_point['img_path']
        batch={
            #'pixel_values':pixel_values,
            'name':name,
            'idx':idx,
            'img_path': img_path,
            'full_prompt':data_point['full_prompt']
        }
        return batch
    
class DataCollatorForPosNegDataset(object):
    """Code from LLaVA Project"""

    def __init__(self,model_name,tokenizer,img_processor, bunny_config=None, bunny_dtype=None):
        self.tokenizer=tokenizer
        self.img_processor=img_processor
        self.model_name=model_name

        self.model_config=bunny_config
        self.model_dtype=bunny_dtype
        
        self.sep="\n"
        self.agent="ASSISTANT"
        self.user="USER"
        if model_name=='Qwen-VL-Chat':
            self.roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}
            system_message="You are a helpful assistant."
            self.im_start = tokenizer.im_start_id
            self.im_end = tokenizer.im_end_id
            self.nl_tokens = tokenizer('\n').input_ids
            self._system = tokenizer('system').input_ids + self.nl_tokens
            self.system_tokens = [self.im_start] + self._system + tokenizer(system_message).input_ids + [self.im_end] + self.nl_tokens
            self._user = tokenizer('user').input_ids + self.nl_tokens
            self._assistant = tokenizer('assistant').input_ids + self.nl_tokens
            

    def generate_input_ids_label(self, full_prompt):
        input_ids = torch.stack([tokenizer_image_token(full_prompt, self.tokenizer, return_tensors='pt')], dim=0)
        targets = input_ids.clone()
        #print (input_ids)
        """
        Currently only support one round of conversation!
        Mask instructions as well as padding tokens
            The code defaultly focuses only on the respones part (i.e., not training on inputs)
        Currently no EOS token added
        """
        cur_len = 1
        sep=self.sep+self.agent+': '#\nASSISTANT: ;
        total_len = int(targets.ne(self.tokenizer.pad_token_id).sum())
        targets[0,:cur_len] = IGNORE_INDEX
        parts = full_prompt.split(sep)
        parts[0] += sep #"[INST] \n"
        #round_len = len(tokenizer_image_token(full_prompt, self.tokenizer))
        instruction_len = len(tokenizer_image_token(parts[0], self.tokenizer)) - 2 #BOS and EOS
        targets[0,cur_len : cur_len + instruction_len] = IGNORE_INDEX
        return input_ids, targets
    
    def generate_input_ids_label_bunny(self, full_prompt):
        input_ids = torch.stack([tokenizer_image_token(full_prompt, self.tokenizer, 
                                                       image_token_index=-200, 
                                                       return_tensors='pt')], dim=0)
        targets = input_ids.clone()
        #prompt: "USER: <image>\n%s ASSISTANT: %s"
        #print (input_ids)
        """
        Currently only support one round of conversation!
        Mask instructions as well as padding tokens
            The code defaultly focuses only on the respones part (i.e., not training on inputs)
        Currently no EOS token added
        """
        cur_len = 1
        sep=self.agent+': '#ASSISTANT: ;
        total_len = int(targets.ne(self.tokenizer.pad_token_id).sum())
        targets[0,:cur_len] = IGNORE_INDEX
        parts = full_prompt.split(sep)
        parts[0] += sep #"[INST] \n"
        #round_len = len(tokenizer_image_token(full_prompt, self.tokenizer))
        instruction_len = len(tokenizer_image_token(parts[0], self.tokenizer)) - 2 #BOS and EOS
        targets[0,cur_len : cur_len + instruction_len] = IGNORE_INDEX
        return input_ids, targets

    def generate_input_ids_label_qwen(self, full_prompt):
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
        input_ids=[]
        labels=[]
        for instance in instances:
            full_prompt=instance["full_prompt"]
            if self.model_name in ["llava-1.5-7b-hf","llava-1.5-13b-hf"]:
                cur_input_ids,cur_labels=self.generate_input_ids_label(full_prompt)
            elif self.model_name=="Qwen-VL-Chat":
                cur_input_ids,cur_labels=self.generate_input_ids_label_qwen(full_prompt)
            else:
                cur_input_ids,cur_labels=self.generate_input_ids_label_bunny(full_prompt)
            input_ids.append(cur_input_ids.squeeze(0))
            labels.append(cur_labels.squeeze(0))
        input_ids, labels = self.formulate_tensor(input_ids, labels)
        if self.model_name in ["llava-1.5-7b-hf","llava-1.5-13b-hf"]:
            pixel_values=self.img_processor([expand2square(
                Image.open(instance['img_path']).convert('RGB'), 
                tuple(int(x*255) for x in OPENAI_CLIP_MEAN)
            ) 
                                             for instance in instances], 
                                            return_tensors='pt').pixel_values
        elif self.model_name=='Bunny-v1_0-3B':
            pixel_values=[self.img_processor([Image.open(instance['img_path']).convert('RGB')], 
                                            self.model_config).to(dtype=self.model_dtype)  
                          for instance in instances
                         ]
            pixel_values=torch.cat(pixel_values,dim=0)
        elif self.model_name=="Qwen-VL-Chat":
            pixel_values=None
            
        batch = {
            "input_ids":input_ids,
            "labels":labels,
            "attention_mask":input_ids.ne(self.tokenizer.pad_token_id),
            'pixel_values':pixel_values
        }

        # print(input_ids.shape, input_ids.device, batch['images'].shape, batch['images'].device)
        if "name" in instances[0]:
            batch['name']=[instance['name'] for instance in instances]
        if "idx" in instances[0]:
            batch['idx']=[instance['idx'] for instance in instances]
        if 'img_path' in instances[0]:
            batch['img_path']=[instance['img_path'] for instance in instances]
        return batch
    
if __name__ == '__main__':
    args=config.parse_opt()
    log_hyperpara(args)
    set_seed(args.SEED)
    device_map = "auto"
    model_name=args.MODEL_NAME.split('/')[-1]
    """
    make sure the consistency of considered aspects of pref data and logp gen data
    Further optimization
        Generate logp for all captions 
        Then, match with logp of captions if the caption is used in preference data
    """
    considered_aspects=args.CONSIDERED_ASPECTS.split(',')

    logp_cls=Pref_Pair_Data(args)
    """
    If I'm not wrong, the data_collator acts like:
        1) each inputs: 2, L (labels the same)
    """

    model_config=None
    model_dtype=None
    if model_name in ["llava-1.5-7b-hf","llava-1.5-13b-hf"]:
        processor = AutoProcessor.from_pretrained(args.MODEL_NAME)

        model = LlavaForConditionalGeneration.from_pretrained(
            args.MODEL_NAME,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
            )
        tokenizer=processor.tokenizer
        image_processor=processor.image_processor
    elif model_name=='Qwen-VL-Chat':
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.generation import GenerationConfig
        tokenizer = AutoTokenizer.from_pretrained(args.MODEL_NAME, 
                                                  padding_side="right",
                                                  use_fast=False,
                                                  trust_remote_code=True
                                                 )
        model = AutoModelForCausalLM.from_pretrained(args.MODEL_NAME, device_map="auto",
                                                     torch_dtype=torch.float16, 
                                                     trust_remote_code=True)
        tokenizer.pad_token_id = tokenizer.eod_id
        image_processor=None

    model.train(False)
    model_dtype=model.dtype
    data_collator = DataCollatorForPosNegDataset(model_name, tokenizer=tokenizer, img_processor=image_processor,
                                                 bunny_config=model_config, bunny_dtype=model_dtype)
    dataloader = torch.utils.data.DataLoader(logp_cls, batch_size=args.infer_batch_size, collate_fn=data_collator,
                                             num_workers=5, shuffle=False)

    print ('Starting to compute log prob for each pos/neg instance')
    logp_info_total={}

    with torch.inference_mode():
        idx=0
        for num_bt,batch in enumerate(dataloader):
            if num_bt >0 and num_bt%50==0:
                print ('Finished iteractions %s, totoal iterations: %s' % (num_bt, len(dataloader)))
                pkl.dump(logp_info_total,open("../pos_neg_pair_gen/all_logp_"+'_'.join([args.DATA_NAME,model_name])+".pkl",'wb'))
            indexs=batch['idx']
            for idx in indexs:
                logp_info_total[idx]={
                    'logp':None,
                    'avg_logp':None,
                    'per_token_logp':None
                }
            img_path=batch['img_path']
            if model_name in ["llava-1.5-7b-hf","llava-1.5-13b-hf"]:
                output = model.forward(
                    input_ids=batch['input_ids'].to(model.device),
                    attention_mask=batch['attention_mask'].to(model.device),
                    pixel_values=batch['pixel_values'].to(model.device),
                    labels=None,
                )
            elif model_name=='Qwen-VL-Chat':
                if args.DEBUG:
                    print (batch['input_ids'][0])
                    print (batch['labels'][0])
                with torch.no_grad():
                    output = model.forward(
                        input_ids=batch['input_ids'].to(model.device),
                        attention_mask=batch['attention_mask'].to(model.device),
                        labels=None,
                    )
            else:
                #print (batch['pixel_values'].shape)
                output = model.forward(
                    input_ids=batch['input_ids'].to(model.device),
                    attention_mask=batch['attention_mask'].to(model.device),
                    images=batch['pixel_values'].to(model.device),
                    labels=None
                    #labels=batch['labels'].to(model.device),
                )[0]
                #print (batch['input_ids'][0])
                #print (batch['attention_mask'][0])
            output.logits=output.logits[:,-batch['labels'].shape[1]:]
            per_token_logp, log_prob, average_log_prob = get_batch_logps(output.logits, batch['labels'].to(model.device), 
                                                                         return_all=True) #B,L; B,; Average means averaging over number of tokens
            per_token_logp=per_token_logp.cpu()
            log_prob=log_prob.cpu()
            average_log_prob=average_log_prob.cpu()
            num_instance=batch['input_ids'].shape[0]
                
            for num,idx in enumerate(indexs):
                logp_info_total[idx]['logp']=log_prob[num].item()
                logp_info_total[idx]['avg_logp']=average_log_prob[num].item()
                logp_info_total[idx]['per_token_logp']=per_token_logp[num]
                if args.DEBUG:
                    print (log_prob[num].item())
                    print (average_log_prob[num].item())
                    print (per_token_logp[num],'\n')

    print ('Saving all logp information!')
    pkl.dump(logp_info_total,open("../pos_neg_pair_gen/all_logp_"+'_'.join([args.DATA_NAME,model_name])+".pkl",'wb'))
