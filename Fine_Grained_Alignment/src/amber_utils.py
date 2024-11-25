import json
import os
import pickle as pkl
import random
from collections import defaultdict
import config

def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data

def load_json(path):
    data=json.load(open(path,'r'))
    return data
    
import nltk
from nltk.stem import WordNetLemmatizer
def extract_nouns(text):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    nouns = [lemmatizer.lemmatize(word) for word, pos in tagged if pos.startswith('NN')]
    return nouns
    
import spacy
nlp = spacy.load("en_core_web_lg")
def check_synonyms_word(word1, word2, similarity_score):
    token1 = nlp(word1)
    token2 = nlp(word2)
    similarity = token1.similarity(token2)
    return similarity > similarity_score

args=config.parse_opt()
association = json.load(open(os.path.join(args.VQA_PATH,'AMBER/data/relation.json'), 'r', encoding='utf-8'))
hallucination_words = []
for word1 in association.keys():
    hallucination_words.append(word1)
    for word2 in association[word1]:
        hallucination_words.append(word2)
            
global_safe_words = []
with open(os.path.join(args.VQA_PATH,'AMBER/data/safe_words.txt'), 'r', encoding='utf-8') as safe_file:
    for line in safe_file:
        line = line.split('\n')[0]
        global_safe_words.append(line)
        
amber=load_json(os.path.join(args.VQA_PATH,'AMBER/data/query/query_generative.json'))
anno_file=load_json(os.path.join(args.VQA_PATH,'AMBER/data/annotations.json'))
similarity_score=0.8
        
def amber_eval(all_pred):
    chair_score=0.0
    chair_num=0.0
    safe_cover_score=0.0
    safe_cover_num=0.0
    hallu_cover_score=0.0
    hallu_cover_num=0.0

    non_hallu_score=0.0
    non_hallu_num=0.0

    names=list(all_pred.keys())
    for i,name in enumerate(names):
        row=amber[int(name)-1]
        pred=all_pred[name]
        #print (i,pred)
        nouns = extract_nouns(pred)
        after_process_nouns = []
        for noun in nouns:
            if noun in hallucination_words:
                after_process_nouns.append(noun)
    
        safe_words = []
        safe_list = []
        for idx, word in enumerate(anno_file[int(name)-1]['truth']):
            safe_words += association[word]#synonyms
            safe_list += [idx] * len(association[word])

        ha_words = []
        ha_list = []
        for idx, word in enumerate(anno_file[int(name)-1]['hallu']):
            ha_words += association[word]
            ha_list += [idx] * len(association[word])
            
        safe_words += anno_file[int(name)-1]['truth']
        safe_len = len(anno_file[int(name)-1]['truth'])
        safe_list += [0] * safe_len
        safe_flag_list = [0] * len(after_process_nouns)
            
        ha_words += anno_file[int(name)-1]['hallu']
        ha_len = len(anno_file[int(name)-1]['hallu'])
        ha_list += [0] * ha_len
        for idx, noun in enumerate(after_process_nouns):
            if noun in global_safe_words:
                continue
                
            if noun in safe_words:
                for j in range(len(safe_words)):
                    if noun == safe_words[j]:
                        if j < (len(safe_list) - safe_len):
                            safe_list[safe_list[j] + len(safe_list) - safe_len] = 1
                        else:
                            safe_list[j] = 1
                        break
                continue
                
            if noun in ha_words:
                for j in range(len(ha_words)):
                    if noun == ha_words[j]:
                        if j < (len(ha_list) - ha_len):
                            ha_list[ha_list[j] + len(ha_list) - ha_len] = 1
                        else:
                            ha_list[j] = 1
                        break
                
            for j, check_word in enumerate(ha_words):
                if check_synonyms_word(noun, check_word, similarity_score):
                    if j < (len(ha_list) - ha_len):
                            ha_list[ha_list[j] + len(ha_list) - ha_len] = 1
                    else:
                        ha_list[j] = 1
                    break
                
            flag = False
            for j, check_word in enumerate(safe_words):
                if check_synonyms_word(noun, check_word, similarity_score):
                    flag = True
                    if j < (len(safe_list) - safe_len):
                        safe_list[safe_list[j] + len(safe_list) - safe_len] = 1
                    else:
                        safe_list[j] = 1
                        break
            if flag == True:
                continue
            
            safe_flag_list[idx] = 1
        chair_score += sum(safe_flag_list)
        chair_num += len(safe_flag_list)
        safe_cover_score += sum(safe_list[-safe_len:])
        safe_cover_num += len(safe_list[-safe_len:])
        hallu_cover_score += sum(ha_list[-ha_len:])
        hallu_cover_num += len(ha_list[-ha_len:])
        if sum(safe_flag_list) == 0:
            non_hallu_score += 1
        non_hallu_num += 1
        
    CHAIR = round(chair_score / chair_num * 100, 1)
    Cover = round(safe_cover_score / safe_cover_num * 100, 1)
    Ha = round(hallu_cover_score / hallu_cover_num * 100, 1)
    Ha_p = round(100 - non_hallu_score / non_hallu_num * 100, 1)
    print("Generative Task:")
    print("CHAIR:\t\t", CHAIR)
    print("Cover:\t\t", Cover)
    print("Hal:\t\t", Ha_p)
    print("Cog:\t\t", Ha, "\n")
    return CHAIR , Cover, Ha_p, Ha

if __name__=='__main__':
    pred_file_dir='../results/dpo_generation_llava1.5/object-object-count-attribute-spatial-scale-text'
    
    all_pred=load_pkl(os.path.join(pred_file_dir,'amber_NUM_113_step_4800_bz_8.pkl'))
    print ('Length of generation:',len(all_pred))
    CHAIR , Cover, Ha_p, Ha=amber_eval(all_pred)