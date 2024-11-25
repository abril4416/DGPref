import os
import torch
import numpy as np
import transformers
import random
import json
import pickle as pkl
import config

from peft import (  # noqa: E402
    LoraConfig,
    PeftModel,
    get_peft_model,
    load_peft_weights,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from peft.tuners.lora import LoraLayer
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import requests

from utils import find_all_linear_names, print_trainable_parameters

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def log_hyperpara(opt):
    dic = vars(opt)
    for k,v in dic.items():
        print(k + ' : ' + str(v))
        
if __name__=='__main__':
    args=config.parse_opt()
    log_hyperpara(args)
    set_seed(args.SEED)
    device_map = "auto"
    model_name=args.MODEL_NAME.split('/')[-1]
    #tokenizer initialization
    #image_processor, tokenizer
    """
    line 109: https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/models/llava/processing_llava.py#L39
    pixel_values = self.image_processor(images, return_tensors=return_tensors)["pixel_values"]
    text_inputs = self.tokenizer(
            text, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length
        )
    """
    bunny_config=None
    model_dtype=None
    if args.MODEL_NAME in ["llava-hf/llava-1.5-7b-hf","llava-hf/llava-1.5-13b-hf"]:
        processor = AutoProcessor.from_pretrained(args.MODEL_NAME)
        processor.tokenizer.pad_token = processor.tokenizer.unk_token
        #model initialization
        if args.load_8bit:
            model = LlavaForConditionalGeneration.from_pretrained(
                args.MODEL_NAME,
                load_in_8bit=args.load_8bit,
                torch_dtype=torch.float16,
                device_map=device_map,
                trust_remote_code=True,
            )
        else:
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
        model = AutoModelForCausalLM.from_pretrained(args.MODEL_NAME, 
                                                     device_map="auto",
                                                     torch_dtype=torch.float16, 
                                                     trust_remote_code=True)
        tokenizer.pad_token_id = tokenizer.eod_id
        image_processor=None
    """
    For different LLaVA versions, the operation is different
        Line 902 in LLaVA/llava/train/train.py
    Padding side already set to be left
    """
    #optimization initialization
    model_dtype=model.dtype
    print('Data type:',model_dtype)
    if args.MODEL_NAME in ["llava-hf/llava-1.5-7b-hf","llava-hf/llava-1.5-13b-hf"]:
        model = prepare_model_for_int8_training(
            model, 
            use_gradient_checkpointing=args.use_gradient_checkpointing)
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()  
    elif args.MODEL_NAME=='Qwen/Qwen-VL-Chat':
        """
        Here set the vision encoder frozen ==> following Qwen-VL github 
        """
        model.transformer.visual.requires_grad_(False)
        if hasattr(model.transformer.visual,'attn_pool'):
            model.transformer.visual.attn_pool.requires_grad_(True)
        model = prepare_model_for_int8_training(
            model, 
            use_gradient_checkpointing=args.use_gradient_checkpointing)
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["c_attn", "attn.c_proj", "w1", "w2"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()  
    #print (model)
    #dataset construction
    """
    Source: a data point from sources
        "conversation": a list
            "from": "gpt/human" #the instructional data, first from human (fake) proposing a query
            "value": [response]
    """
    if args.dpo_training:
        from posneg_pair_dataset import PosNeg_Pair_Data
        train_cls=PosNeg_Pair_Data(args,tokenizer,mode='train',dataset=args.DATA_NAME)
        if args.DATA_NAME=='vsg':
            val_cls=PosNeg_Pair_Data(args,tokenizer,mode='val',dataset=args.DATA_NAME)
        else:
            val_cls=PosNeg_Pair_Data(args,tokenizer,mode='val',dataset='vsg')
        print ('Length of validation:',len(val_cls))
    else:
        """
        this is deprecated: no implementation!!!
        Just leave it there!!!
        """
        train_cls=HM_Data(args,processor,mode='train')
        val_cls=HM_Data(args,processor,mode='test')
    """
    DEBUG RELATED
    from torch.utils.data import DataLoader
    train_loader=DataLoader(train_cls,
                            1,
                            shuffle=True,
                            num_workers=2)
    for i,batch in enumerate(train_loader):
        print (batch['labels'])
        print (natch['input_ids'])
    """
    if args.dpo_training:
        from posneg_pair_dataset import DataCollatorDPODataset
        data_collator =DataCollatorDPODataset(model_name,tokenizer,image_processor,
                                              bunny_config, model_dtype)
    else:
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    if args.dpo_training:
        from dpo_train import train_for_epochs
        train_for_epochs(model,tokenizer,image_processor,args,  
                         data_collator,
                         train_cls,val_cls)
    else:
        from train import train_for_epochs
        train_for_epochs(model,processor,args,  
                         data_collator,
                         train_cls,val_cls)