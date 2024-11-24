# DGPref: Implementation for [Decompose and Leverage Preferences from Expert Models for Improving Trustworthiness of MLLMs](https://arxiv.org/pdf/2411.13697)

## Overview

## Content
- [Dataset Preparation](#dataset-preparation)
- [Experiment Setting](#experiment-setting)
- [Preference Data Generation](#preference-data-generation)
- [Model Training](#model-training)

## Dataset Preparation
In order to replicate the code and performance, you need to prepare two-fold of data: 1) data for generating preference data; and 2) data for evaluation.
### Data for Preference Data Generation
Our code is compatible with any images (i.e., you can generate raw detailed descriptions for any images and generate preference data with our code). Specifically, in our implementation, we use the [Visual Genome images](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html), as it has some human annotations, which facilitates in ablation studies. The Visual Genome dataset is denoted as vsg in our implementation. You can download the Visual Genome images with the script below:
```
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip -P images_part_1
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip -P images_part_2
```
Scene graphs for Visual Genome images are available [here](https://cs.stanford.edu/people/dorarad/gqa/download.html), which we denoted as train_sceneGraphs/val_sceneGraphs.

Theoritically, we can use any images for preference data generation. An alternative option is to use COCO-2014 training images for preference data generation. The dataset can be downloaded with the script:
```
wget images.cocodatsaet.org/zips/train2014.zip
```

### Data for Evaluation
We use three datasets for evaluation:
1. ObjectHal Bench (which we denoted with objhal in our implementation). We follow previous works to use the exact 300 examples in the original ObjectHal bench for evaluation. You can access the evaluation dataset (as well as the input prompts) [here](https://github.com/RLHF-V/RLAIF-V/blob/main/eval/data/obj_halbench_300_with_image.jsonl) ObjectHal dataset uses images from COCO-2014 (val split) You can download the images via [this link](https://cocodataset.org/#download) Optionally, you can use the script below for dowloading:
```bash
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip
```
2. MMHal Bench (which we denoted with mmhal in our implementation). You can download the data [here](https://github.com/RLHF-V/RLAIF-V/blob/main/eval/data/mmhal-bench_answer_template.json). Images for MMHal dataset are availabel in [this link](https://drive.google.com/file/d/1mQyAbeGgRyiVV6qjVkUI1uY_g9E-bDTH/view0).
3. AMBER (which we denoted with amber in our implementation). You can access all information for AMBER via [this link](https://github.com/junyangwang0410/AMBER) The data is also available via the link.

**Be careful!!!! After downloading all data, you still need to update the path in our implementation to your file path.**

## Experiment Setting
Essential requirements are listed in [essential_requirement.txt](https://github.com/abril4416/DGPref/blob/main/essential_requirement.txt), the versions of which matter. For other required packages, please see the import command in each file.

## Preference Data Generation
1. Generate raw data from MMLMs without any further training; Execute:
   ```
   python Fine_Grained_Alignment/src/raw_caption_gen.py
   ```
   Please also check what dataset you would like to generate raw captions for (You need to check *val_dataset_names* in the [code](https://github.com/abril4416/DGPref/blob/main/Fine_Grained_Alignment/src/raw_caption_gen.py) for details). 
2. Decompose raw outpus from MMLMs accoding to different aspects (object, attributes, relations, etc). The decomposition is conducted with in-context learning by providing a few demonstrations. The demonstration templates can be found [here](https://github.com/abril4416/DGPref/tree/main/Decomp_Gen_Cap/templates_llama); Execute:
   ```
   python Decomp_Gen_Cap/src/meta_llama3_extract.py --GQA_PATH [YOUR_FILE_PATH] --target_file [YOU_RAW_CAPTION_PATH]
   ```
3. Assign decomposed results to proper experts for evaluation and generate expert scores; Execute:
   ```
   python Decomp_Gen_Cap/src/program_to_tool.py
   ```
   Make sure your paths are set corretly.
4. Use the expert scores to generate paired preferred and rejected data; Execute:
   ```
   python Fine_Grained_Alignment/src/pref_data_gen.py --CONSIDERED_ASPECTS [YOUR_CONCERNED_ASPECTS]
   ```
   As detailed image descriptions focus on different aspects (e.g, object, attributes, relations, etc.). The preference data can also be generated with certain aspects (i.e., prefer captions better at certain aspects). You can set the aspects to focus on with *--CONSIDERED_ASPECTS*. Meanwhile,the scoring of each caption can be a summation of scores from expert evaluators or averaging over scores of expert evaluators (with *--AVG_ADV_SCORES*).
   
## Model Training
After generating the preference data, you can conduct direct preference optimization (DPO) of MLLMs. To conduct DPO, the log probabilities of raw captions are also needed as the reference log probability. You need to execute:
```
python Fine_Grained_Alignment/src/logp_gen_for_pairs.py
```
for generating the log probabilities.
Then, DPO training can be conducted with:
```
python Fine_Grained_Alignment/src/main.py --micro_batch_size 4 --SAVE_NUM 4 --gen_step 1500 --gen_start 300 --eval_step 1500 --lora_r 8 --learning_rate 5e-7  --num_epochs 6 --batch_size 8  --CONSIDERED_ASPECTS [YOUR_CONCERNED_ASPECTS] --HAL_DATASETS 'amber,mmhal,objhal' 
```
**You can always use the DEBUG mode (set --DEBUG True) for easier debugging~**
