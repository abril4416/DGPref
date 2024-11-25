import argparse 

def parse_opt():
    parser=argparse.ArgumentParser()
    parser.add_argument('--VQA_PATH',
                        type=str,
                        default='/mnt/data1/rui/Rui_Data_Space/VQA')
    #for VQA related data, point out the dataset name
    parser.add_argument('--DATA_NAME',
                        type=str,
                        default='vsg')#vizwiz, iconqa, text-vqa,visual-spatial-reasoning
    parser.add_argument('--GEN_DATASETS', type=str, 
                        default="vsg")
    """
    Here can add vsg, for evaluation
    """
    parser.add_argument('--HAL_DATASETS',
                        type=str,
                        default='objhal,mmhal,amber')#vizwiz, iconqa, text-vqa,visual-spatial-reasoning
    #llava-hf/llava-1.5-7b-hf, BAAI/Bunny-v1_0-3B
    parser.add_argument('--MODEL_NAME',
                        type=str,
                        default='Qwen/Qwen-VL-Chat')#blip-2,instruct-blip,mplug-owl,llava
    parser.add_argument('--SEED',
                        type=int,
                        default=1111)#blip-2,instruct-blip,mplug-owl,llava
    parser.add_argument('--SAVE_NUM', type=int, 
                        default=0)
    parser.add_argument('--INFERENCE_BATCH', type=int, 
                        default=12)

    parser.add_argument('--NEG_WORD', type=str, 
                        default='Yes')
    parser.add_argument('--POS_WORD', type=str, 
                        default='No')
    parser.add_argument('--MAX_LEN', type=int,
                        default=100)
    #for ablation studies ==> scale of the dataset
    parser.add_argument('--HUMAN_SIZE', type=int, 
                        default=-1)
    """
    Consider all aspects as default
        can selectively use subset of aspects
        currently, simple summation
    """
    parser.add_argument('--CONSIDERED_ASPECTS', type=str, 
                        default="object,count,attribute,spatial,scale,text")
    parser.add_argument('--PUNISH_SHORT', type=bool, 
                        default=False)
    parser.add_argument('--HALF_VSG', type=bool, 
                        default=False)
    parser.add_argument('--USE_BOTH', type=bool, 
                        default=False)
    #dpo related args
    """
    dpo_use_average
    dpo_token_weighted
    dpo_beta 0.1
    """
    parser.add_argument('--dpo_training', type=bool, 
                        default=True)
    parser.add_argument('--dpo_use_average', type=bool, 
                        default=False)
    parser.add_argument('--dpo_token_weighted', type=bool, 
                        default=False)
    parser.add_argument('--dpo_token_weight', type=float, 
                        default=1.0)
    parser.add_argument('--dpo_beta', type=float, 
                        default=0.1)
    
    #optimization related
    parser.add_argument('--load_8bit', type=bool, 
                        default=False)
    parser.add_argument('--batch_size', type=int, 
                        default=8)
    parser.add_argument('--infer_batch_size', type=int, 
                        default=6)
    parser.add_argument('--micro_batch_size', type=int, 
                        default=1)
    """
    A bug while not solved yet: 
        num_iterations % logging_step should not be 0!!!!
    """
    parser.add_argument('--logging_steps', type=int, 
                        default=68)
    parser.add_argument('--save_total_limit', type=int, 
                        default=5)
    parser.add_argument('--warmup_steps', type=int, 
                        default=100)
    parser.add_argument('--num_epochs', type=int, 
                        default=4)
    parser.add_argument('--learning_rate', type=float, 
                        default=5e-7)
    parser.add_argument('--cutoff_len', type=int, help='Maximum sequence length to process.',
                        default=256)
    parser.add_argument('--use_gradient_checkpointing', type=bool, 
                        default=False)
    parser.add_argument('--group_by_length', type=bool, 
                        default=False)
    parser.add_argument('--fp16', type=bool, 
                        default=True)
    
    parser.add_argument('--init_from_checkpoint', type=bool, 
                        default=False)
    parser.add_argument('--lora_checkpoint_dir', type=str, 
                        default="/mnt/data1/rui/LoRA/Modules")
    parser.add_argument('--lora_checkpoint_file', type=str, 
                        default="checkpoint-60300")
    

    #LoRA related
    parser.add_argument('--lora_r', type=int, default=8,
                        help='curvature.')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='The initialization coefficient of lora-alpha.')  
    parser.add_argument('--lora_dropout', type=int, default=0.05,
                        help='The initialization coefficient of lora_dropout.')
    parser.add_argument('--target_modules', type=str, 
                        default=r'.*language_model.*\.(q_proj|v_proj)')    

    #generation related
    parser.add_argument('--temperature', type=float, 
                        default=0.7)
    parser.add_argument('--top_p', type=float, 
                        default=0.75)
    parser.add_argument('--top_k', type=int, 
                        default=40)
    parser.add_argument('--num_beams', type=int, 
                        default=4)
    parser.add_argument('--max_new_tokens', type=int, 
                        default=1024)

    #saving and evaluation
    parser.add_argument('--eval_step', type=int, 
                        default=50)
    parser.add_argument('--gen_step', type=int, 
                        default=200)
    parser.add_argument('--gen_start', type=int, 
                        default=350)
    parser.add_argument('--val_set_size', type=int, 
                        default=500)
    parser.add_argument('--save_step', type=int, 
                        default=500)
    parser.add_argument('--output_dir', type=str, 
                        default='/mnt/data1/rui/LoRA/Output')
    parser.add_argument('--resume_from_checkpoint', type=str, 
                         default=None)
    
    parser.add_argument('--DEBUG',
                        type=bool,
                        default=False)
    #using GPT-4 rather than GPT-3.5
    parser.add_argument('--GPT_ADV_VAL',
                        type=bool,
                        default=False)
    """
    Whether using GT object detection annotation
        provided by VG dataset
    """
    parser.add_argument('--GT_OBJ',
                        type=bool,
                        default=False)
    parser.add_argument('--PARTIAL_POSNEG',
                        type=bool,
                        default=False)
    args=parser.parse_args()
    return args