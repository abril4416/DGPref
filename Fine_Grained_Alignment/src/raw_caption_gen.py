import os
import torch
import numpy as np
import transformers
import random
import json
import pickle as pkl
import config

from peft import PeftModel, load_peft_weights, set_peft_model_state_dict, LoraConfig, get_peft_model, prepare_model_for_int8_training
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
from hal_cap_dataset import Hal_Cap_Data

from llava.constants import expand2square,OPENAI_CLIP_MEAN

"""
This implementation is for generating raw captions/answers from MLLMs
    Raw generations will be used for generating DPO data
    Specifically, we decompose and check raw generations and get preference scores
    Preference scores will be used for DPO training
"""

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

def log_hyperpara(opt):
    dic = vars(opt)
    for k,v in dic.items():
        print(k + ' : ' + str(v))

def img2base64(file_name):
    with open(file_name, 'rb') as f:
        encoded_string = base64.b64encode(f.read())
        return encoded_string

if __name__=='__main__':
    args=config.parse_opt()
    log_hyperpara(args)
    set_seed(args.SEED)
    device_map = "auto"
    model_name=args.MODEL_NAME.split('/')[-1]
    if os.path.exists(os.path.join('../results/gen_results/raw_gen',model_name))==False:
        print ('Making a dir:', os.path.join('../results/gen_results/raw_gen',model_name))
        os.mkdir(os.path.join('../results/gen_results/raw_gen',model_name))
        
    #tokenizer initialization
    #image_processor, tokenizer
    if model_name=="llava-1.5-7b-hf":
        processor = AutoProcessor.from_pretrained(args.MODEL_NAME)

        model = LlavaForConditionalGeneration.from_pretrained(
            args.MODEL_NAME,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
            )
    elif model_name== "Bunny-v1_0-3B":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        """
        Disable warnings...
        """
        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()
        import warnings
        warnings.filterwarnings('ignore')
        """
        This is fairly important for Bunny
        Or will have different devices error (cpu vs. cuda)
        """
        torch.set_default_device("cuda")
        model = AutoModelForCausalLM.from_pretrained(
            args.MODEL_NAME,
            torch_dtype=torch.float16, # float32 for cpu
            trust_remote_code=True).cuda()
        tokenizer = AutoTokenizer.from_pretrained(
            args.MODEL_NAME,
            trust_remote_code=True)
    elif model_name=="Qwen-VL-Chat":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.generation import GenerationConfig
        tokenizer = AutoTokenizer.from_pretrained(args.MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.MODEL_NAME, device_map="auto",
                                                     torch_dtype=torch.float16, 
                                                     trust_remote_code=True)
        
    if args.init_from_checkpoint:
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
        lora_path=os.path.join(args.lora_checkpoint_dir,args.lora_checkpoint_file)
        print ('LoRA:',lora_path)
        adapters_weights = load_peft_weights(lora_path)
        set_peft_model_state_dict(model, adapters_weights)
        #model = PeftModel.from_pretrained(model, lora_path, is_trainable=False)   
        #model.load_adapter(lora_path)
    model.eval()
    #val_dataset_names=['mmhal','objhal']
    val_dataset_names=args.GEN_DATASETS.split(',')
    do_sample=(args.temperature>0.0)
    for name in val_dataset_names:
        if os.path.exists(os.path.join('../results/gen_results/raw_gen',model_name,name+'.pkl')):
            total=load_pkl(os.path.join('../results/gen_results/raw_gen',model_name,name+'.pkl'))
        else:
            total={}
        #total={}
        print ('Generating raw responses for:',name,'Already generated:',len(total))
        if name in ['vsg','coco']:
            val_data=Hal_Cap_Data(args,name,mode='train')
        else:
            val_data=Hal_Cap_Data(args,name)
        for i,row in enumerate(val_data.entities):
            if row['idx'] in total:
                continue
            if i>0 and i%300==0:
                print ('Saving...',i,len(val_data.entities))
                pkl.dump(total,open(os.path.join('../results/gen_results/raw_gen',model_name,name+'.pkl'),'wb'))
            prompt=row['prompt']
            im=Image.open(row['img_path']).convert('RGB')
            if model_name in ["llava-1.5-7b-hf","llava-1.5-13b-hf"]:
                #if name in ['gqa','text-vqa','vizwiz']:
                #    im = expand2square(im, tuple(int(x*255) for x in OPENAI_CLIP_MEAN))
                im = expand2square(im, tuple(int(x*255) for x in OPENAI_CLIP_MEAN))
                inputs = processor(text=prompt, images=im, return_tensors="pt")
                #generate_ids = model.generate(**inputs.to(model.device), max_new_tokens=128)
                """
                To enhance the GPU utilization percentage
                    Batch inference is highly recommended!!!! ==> the current version only supports per instance inference
                    It is slow
                Bad thing is: results from batch inferences differ from single instance inference
                """
                generate_ids = model.generate(**inputs.to(model.device),
                                              do_sample=do_sample,
                                              temperature=args.temperature,
                                              max_new_tokens=1024)
                results=processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                #print (results)
                results=results.split('ASSISTANT: ')[-1]
            elif model_name=='Bunny-v1_0-3B':
                text_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]
                input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1], dtype=torch.long).unsqueeze(0).to(model.device)
                image_tensor = model.process_images([im], model.config).to(dtype=model.dtype, device=model.device)
            
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.half().to(model.device),
                    do_sample=do_sample,
                    temperature=args.temperature,
                    max_new_tokens=1024
                )[0]
                results=tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
            elif model_name=="Qwen-VL-Chat":
                query = tokenizer.from_list_format([{'image': row['img_path']},
                                                    {'text': prompt},
                                                   ])
                results, history = model.chat(tokenizer, query=query, history=None)
            total[row['idx']]=results
            if args.DEBUG:
                print (row['idx'])
                print ('\t',results)
        print ('Saving...')
        pkl.dump(total,open(os.path.join('../results/gen_results/raw_gen',model_name,name+'.pkl'),'wb'))
