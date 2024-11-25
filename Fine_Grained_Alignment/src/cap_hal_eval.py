import os
import torch
import numpy as np
import transformers
import random
import json
import pickle as pkl
import config
from collections import defaultdict

from PIL import Image
from hal_cap_dataset import Hal_Cap_Data, load_json, load_pkl
from chair_utils import get_mscoco_obj, caption_to_words, get_node_obj_set

from llava.constants import expand2square, OPENAI_CLIP_MEAN, process_anyres_image
from amber_utils import amber_eval

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def cap_gen_val(model,tokenizer,image_processor,args,training_step):
    val_dataset_names=args.HAL_DATASETS.split(',')
    considered_aspects='-'.join(args.CONSIDERED_ASPECTS.split(','))
    if os.path.exists(os.path.join('../results/dpo_generation_llava1.5',considered_aspects))==False:
        os.mkdir(os.path.join('../results/dpo_generation_llava1.5',considered_aspects))
    print (val_dataset_names, len(val_dataset_names))
    model_name=args.MODEL_NAME.split('/')[-1]
    if args.MODEL_NAME in ["llava-hf/llava-1.5-7b-hf","llava-hf/llava-1.5-13b-hf"]:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(args.MODEL_NAME)
        processor.tokenizer.pad_token = processor.tokenizer.unk_token
    chair_is=[]
    chair_ss=[]
    do_sample=(args.temperature>0.0)
    for dataset_name in val_dataset_names:
        val_data=Hal_Cap_Data(args,dataset_name)
        if dataset_name=='vsg':
            val_data.entities=val_data.entities[:100] #use the first 1000 cases for generation testing
        if args.DEBUG:
            val_data.entities=val_data.entities[:18] 
        total={}
        if dataset_name in ['amber'] and training_step % (args.eval_step*2)!=0:
            """
            AMBER is too many
            Validate less frequent
            """
            chair_is.append(0.0)
            chair_ss.append(0.0)
            continue
        name='_'.join([dataset_name,'NUM',str(args.SAVE_NUM),'step',str(training_step),'bz',str(args.batch_size)])
        #entity_chunks=list(chunks(val_data.entities,args.infer_batch_size))
        print ('Number of chunks for ',name,": ",len(val_data.entities))
        for i,row in enumerate(val_data.entities):
            if i>0 and i%300==0:
                print ('Saving...',i,len(val_data.entities))
                pkl.dump(total,open(os.path.join('../results/dpo_generation_llava1.5',considered_aspects,name+'.pkl'),'wb'))
            prompt=row['prompt'] 
            im=Image.open(row['img_path']).convert('RGB')
            if model_name in ["llava-1.5-7b-hf","llava-1.5-13b-hf"]:
                #if name in ['gqa','text-vqa','vizwiz']:
                #    im = expand2square(im, tuple(int(x*255) for x in OPENAI_CLIP_MEAN))
                if dataset_name=='mmhal':
                    im=process_anyres_image(im)
                else:
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
            total[row['raw_img_id']]=results
            if args.DEBUG:
                print (i)
                print ('Generated outputs:',results)
            #total[row['raw_img_id']]=results
        pkl.dump(total,open(os.path.join('../results/dpo_generation_llava1.5',considered_aspects,name+'.pkl'),'wb'))
        #computation for hallucination rate
        #preparations
        if dataset_name not in ['vsg', 'objhal','amber']:
            """
            No implementation of CHAIR evaluation for other datasets
            """
            chair_is.append(0.0)
            chair_ss.append(0.0)
            continue
        if dataset_name=='vsg':
            val_graphs=load_json(
                os.path.join(args.VQA_PATH,'GQA/val_sceneGraphs.json')
                )
        elif dataset_name=='objhal':
            print ('Loading from objectHal bench')
            imid_to_objects=load_pkl(os.path.join(args.VQA_PATH,'Hal_Benchs/total_imgid_to_objects.pkl'))
            print ('Length of image to id:',len(imid_to_objects))
        if dataset_name=='amber':
            CHAIR , Cover, Ha_p, Ha=amber_eval(total)
            chair_is.append(CHAIR)
            chair_ss.append(Cover)
            continue
            """
            anno_file=load_json(os.path.join(args.VQA_PATH,'AMBER/data/annotations.json'))
            anno_objects={str(row["id"]):row["truth"] for row in anno_file}
            """
        #print (len(val_graphs))
        mscoco_objects, inverse_synonym_dict=get_mscoco_obj(args.VQA_PATH)
        chair_i=0.0
        chair_s=0.0
        coco_word_count=0.0
        hall_word_count=0.0
        for idx in total:
            gen_cap=total[idx]

            if dataset_name=='vsg':
                graph_idx=idx
                info=val_graphs[graph_idx]
                obj_infos=info['objects']
                ids_to_names={}
                img_obj=defaultdict(int)

                for obj_idx in obj_infos:
                    obj_info=obj_infos[obj_idx]
                    obj_name=obj_info['name']
                    ids_to_names[obj_idx]=obj_name
                for obj_idx in obj_infos:
                    obj_info=obj_infos[obj_idx]
                    obj_name=obj_info['name']
                    img_obj[obj_name]+=1
            
                gt_obj=[o for o in img_obj]
                if args.DEBUG:
                    print (gt_obj)
                gt_obj=get_node_obj_set(gt_obj, mscoco_objects, inverse_synonym_dict)
            elif dataset_name=='objhal':
                gt_obj=imid_to_objects[int(idx)]
            elif dataset_name=='amber':
                gt_obj=anno_objects[idx]
                gt_obj=get_node_obj_set(gt_obj,mscoco_objects,inverse_synonym_dict)
            words, node_words=caption_to_words(gen_cap, mscoco_objects, inverse_synonym_dict)
            """
            penalize if repeating the hallucinated objects multiple times
            """
            num_pred_obj=len(node_words)
            """
            Object-level hallucinations
            Lacks implementations for sentence-level hallucinations
            """
            num_hallucinated=len([obj for obj in node_words if obj not in gt_obj])
            hall_word_count+=num_hallucinated
            coco_word_count+=num_pred_obj
            if args.DEBUG:
                print (num_hallucinated,num_pred_obj)
            if num_hallucinated>0:
                chair_s+=1
            if args.DEBUG:
                print ('Generated Caption:',gen_cap)
                print ('GT objects:',gt_obj)
                print ('\tHallucinated objects:',num_hallucinated)
        chair_i=hall_word_count*100.0/coco_word_count
        chair_s=chair_s/len(total)*100.0
        chair_is.append(chair_i)
        chair_ss.append(chair_s)
        if args.DEBUG:
            print (len(total))
            print ('Instance level hallucination:',chair_i)
            print ('Sentence level hallucination:',chair_s)
    return chair_is, chair_ss