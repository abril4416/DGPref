import os
import torch
import numpy as np
import random
import json
import pickle as pkl

import config

#from omnilmm.model.omnilmm import OmniLMMForCausalLM
#from omnilmm.model.utils import build_transform
from transformers import AutoTokenizer, AutoModel
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def init_omnilmm(opt):
    torch.backends.cuda.matmul.allow_tf32 = True
    #disable_torch_init()
    model_name=opt.MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, model_max_length=opt.max_new_tokens)
    model = OmniLMMForCausalLM.from_pretrained(
        model_name, tune_clip=True, torch_dtype=torch.bfloat16
        ).to(device='cuda', dtype=torch.bfloat16)
    image_processor = build_transform(
        is_train=False, 
        input_size=model.model.config.image_size, 
        std_mode='OPENAI_CLIP')
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    assert mm_use_im_start_end

    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IM_END_TOKEN], special_tokens=True)


    vision_config = model.model.vision_config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
        [DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids(
        [DEFAULT_IM_START_TOKEN, 
         DEFAULT_IM_END_TOKEN])
    image_token_len = model.model.config.num_query
    return model, image_processor, image_token_len, tokenizer

if __name__=='__main__':
    """
    Testing if omnilmm can be loaded properly
    """
    args=config.parse_opt()
    #set_seed(args.SEED)
    model, image_processor, image_token_len, tokenizer=init_omnilmm(args)