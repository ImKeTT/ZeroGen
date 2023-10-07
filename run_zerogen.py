#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
@file: run_zerogen.py
@author: ImKe at 2022/7/16
@email: tuisaac163@gmail.com
@feature: #Enter features here
"""

from model.clip import CLIP
from model.simctg import SimCTG
from model.llm import ZeroGen
from instructions import get_instruction_vqa, get_instruction_caption
from model.topic_classifier import TopicClassifier
import torch, os, sys, collections, json, time
from tqdm import tqdm
from PIL import Image
import argparse
import numpy as np
import random
import pickle
import datetime
from evaluation.pycocoevalcap.eval import COCOEvalCap
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

config = json.load(open("./config.json"))
DATA_DIR = config['DATA_DIR']
CACHE_DIR = config['CACHE_DIR']
KEYWORD_PREFIX_PATH="./wordlists"
LOAD_PREFIX_PATH="./output"
GEN_PREFIX_PATH="./generated"
RESULTS_PREFIX_PATH="./results"

os.makedirs(LOAD_PREFIX_PATH, exist_ok=True)
os.makedirs(RESULTS_PREFIX_PATH, exist_ok=True)


parser = argparse.ArgumentParser()
## data preparation
parser.add_argument("--language_model_name", default="cambridgeltl/magic_mscoco", type=str, required=False)
parser.add_argument("--classifier_name", default=None, type=str, required=False)
parser.add_argument("--task", default="mscoco", type=str, required=False)
parser.add_argument("--clip_name", default="openai/clip-vit-base-patch32", type=str, required=False)
parser.add_argument('--gpu', nargs='+', type=int, default=[0])
parser.add_argument('--n_gpu', type=int, default=1)
parser.add_argument('--seed', type=int, default=3407)
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--workers", type=int, default=3, required=False)

## generation
parser.add_argument('--beam_width', type=int, default=5)
parser.add_argument('--n_word', type=int, default=1)
parser.add_argument('--n_obj', type=int, default=0)
parser.add_argument('--eta', type=float, default=1.0)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument("--k", type=int, default=45, help='k for magic search')
parser.add_argument('--decoding_len', type=int, default=16)
parser.add_argument('--clip_text_max_len', type=int, default=50)

parser.add_argument('--condition_method', type=str,
                    default="add", choices=["add", "mul"])
parser.add_argument('--keyword', type=str, default="legal")
parser.add_argument("--test_path", type=str)
parser.add_argument("--test_obj_path", type=str, default=None)
parser.add_argument("--do_sample", action='store_true')
parser.add_argument("--kw_mode", type=str, default='sum', choices=['max', 'sum', 'mean', 'random'])

## K2T generation
parser.add_argument("--k2t_mode", type=str, default='all', choices=['all', 'random', 'next'])
parser.add_argument("--ct_path", type=str,
                    default=os.path.join(DATA_DIR, '/npy_data/converter_table_glove.npy'))
parser.add_argument("--ed_path", type=str,
                    default=None)
parser.add_argument("--news_label", type=str,
                    default='pos', choices=['pos', 'neg'])
parser.add_argument("--guarantee", action='store_true')
parser.add_argument("--only_max", action='store_true')
parser.add_argument("--k2t", action='store_true')
parser.add_argument("--c2t", action='store_true')
parser.add_argument("--n_topic", type=int, default=3)
parser.add_argument("--alpha_scale", action='store_true')
parser.add_argument("--alpha_dw_mode", type=int, default=1, choices=[1, 2],
                    help="1 for mean pooling weighting, 2 for element-wise weighting")
parser.add_argument("--alpha_activesize", type=float, default=0.2)
parser.add_argument("--alpha_upper", type=float, default=3.0)
parser.add_argument("--beta_scale", action='store_true')
parser.add_argument("--beta_activesize", type=float, default=0.2)
parser.add_argument("--beta_upper", type=float, default=3.0, help="3.0 for filckr or 2.0 for mscoco")
parser.add_argument("--vis_window_len", type=int, default=1)
parser.add_argument("--BOW_top_n", type=int, default=None)
parser.add_argument("--obj4topic", action='store_true')
parser.add_argument("--n_obj4t", type=int, default=1)
parser.add_argument("--cal_sim_once", action='store_true')
parser.add_argument("--update_keywords", action='store_true')
parser.add_argument("--step_gap", type=int, default=1,
                    help="step to update keyword similarity")

parser.add_argument("--device", type=int, default=0)
parser.add_argument("--num_instruction", type=int, default=5)
parser.add_argument("--test_num", type=int, default=1000)
parser.add_argument("--seq2seq", action='store_true', default=False,
                    help="Use FlanT5 if set this flag to True")


def setup_seed(seed=3407):
    os.environ["PYTHONHASHSEED"]=str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.enabled=False

def gen_sent(sent, sos_token, eos_token):
    sent = [s[s.find(sos_token) + len(sos_token):] for s in sent]
    return sent

def integrate_keywords(topic_dict, obj_dict):
    out = {}
    for key in topic_dict.keys():
        out[key] = []
        if len(obj_dict[key]):
            out[key].extend(obj_dict[key][:args.n_obj4t])
        out[key].extend(topic_dict[key])
    return out

def preprocess_topic_words(topic_words_dict, test_lines, n_topic, sample=False):
    out_topic_words = []
    for item in test_lines:
        if sample:
            out_topic_words.append(random.sample(topic_words_dict[item['topic']], n_topic))
        else:
            out_topic_words.append(topic_words_dict[item['topic']][:n_topic])
    return out_topic_words

def get_num_topic_dict():
    ## 'world', 'sports', 'business', 'science'
    num2topic = {"world": 0, "sports": 1, "business": 2, "science": 3}
    topic2num = {0: "world", 1: "sports", 2: "business", 3: "science"}
    return num2topic, topic2num

def get_num_style_dict():
    num2topic = {"romantic": 0, "humor": 1}
    topic2num = {0: "romantic", 1: "humor"}
    return num2topic, topic2num

def zerogen_generate(args):
    now = datetime.datetime.now()
    setup_seed(args.seed)

    def get_prompt_id(text, tokenizer):
        text = text
        tokens = tokenizer.tokenize(text)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.LongTensor(input_ids).view(1, -1)
        return input_ids

    if torch.cuda.is_available():
        print('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    device = args.device

    save_path_prefix = GEN_PREFIX_PATH
    import os
    if os.path.exists(save_path_prefix):
        pass
    else:  # recursively construct directory
        os.makedirs(save_path_prefix, exist_ok=True)

    save_name = "output.json"
    full_save_path = os.path.join(GEN_PREFIX_PATH, save_name)
    print('full save path is {}'.format(full_save_path))

    print('Loading data...')
    with open(args.test_path) as f:
        item_list = json.load(f)

    with open(args.train_path) as f:
        instruction_item_list = json.load(f)

    obj_list = None
    if args.n_obj and args.k2t:
        with open(args.test_obj_path) as f:
            obj_list = json.load(f)
    if args.obj4topic:
        with open(args.obj_test_obj_path) as f:
            obj_list_tmp = json.load(f)
        if obj_list is None:
            obj_list = obj_list_tmp
        else:
            obj_list = integrate_keywords(obj_list, obj_list_tmp)
        args.n_obj += args.n_obj4t

    print('Data loaded.')
    print('Number of test instances is {}'.format(len(item_list)))

    converter_table, enc_dict, mode = None, None, 'random'
    if args.k2t:
        print('Loading K2T Table...')
        converter_table = np.load(args.ct_path)
        with open(args.ed_path, 'rb') as f:
            enc_dict = pickle.load(f)
        mode = args.k2t_mode
        print('K2T Table Loaded...')
    if args.obj4topic:
        print('Loading K2T Table for Object...')
        converter_table = np.load(args.ct_path)
        with open(args.obj_ed_path, 'rb') as f:
            enc_dict_obj = pickle.load(f)
        if enc_dict is not None:
            enc_dict.update(enc_dict_obj)
        else:
            enc_dict = enc_dict_obj
        mode = args.k2t_mode
        print('K2T Table for Object Loaded...')

    print('Loading CLIP...')
    clip = CLIP(args.clip_name, device)
    clip.eval()
    print('CLIP loaded!')

    topic_classifier, num2topic = None, None
    if args.c2t:
        if args.task == "visnews":
            num2topic, _ = get_num_topic_dict()
        else:
            num2topic, _ = get_num_style_dict()
        print('Loading Classifier...')
        topic_classifier = TopicClassifier(args.classifier_name, device)
        topic_classifier.eval()
        print('Classifier loaded!')

    print('Loading off-the-shelf language model...')
    sos_token, pad_token = r'<-start_of_text->', r'<-pad->'
    eos_token = r'<|endoftext|>'
    clip_text_max_len = 60
    generation_model = ZeroGen(args.language_model_name, seq2seq=args.seq2seq,
                                   converter_table=converter_table)
    if cuda_available:
        generation_model = generation_model.to(device)
    generation_model.eval()
    print('Language model loaded.')

    result_list = []
    invalid_num = 0

    print('----------------------------------------------------------------')
    with torch.no_grad():
        test_num = args.test_num # len(item_list)
        # test_num = 10
        print('Number of inference instances is {}'.format(test_num))
        time_list = []

        clip_vis_dict = None
        if args.clip_vis_pkl is not None and os.path.exists(args.clip_vis_pkl):
            with open(args.clip_vis_pkl, 'rb') as f:
                clip_vis_dict = pickle.load(f)

        text_topics = None
        for p_idx in tqdm(range(test_num)):
            # instructions = [random.choice(item['captions']) for item in random.sample(instruction_item_list, args.num_instruction)]
            # instructions = get_instruction_caption(instructions)
            one_test_dict = item_list[p_idx]
            text_obj = None if (obj_list is None or len(obj_list)==0) else obj_list[one_test_dict['image_name']][: args.n_obj]
            one_res_dict = {
                'split': one_test_dict['split'],
                'topic': one_test_dict['topic'] if 'topic' in one_test_dict.keys() else 'none',
                'image_name': one_test_dict['image_name'],
                'captions': one_test_dict['captions'],
                'title': one_test_dict['title'] if 'title' in one_test_dict.keys() else 'none',
                'obj': 'none' if obj_list is None else text_obj
            }

            if args.k2t or args.obj4topic:
                keywords = text_obj
                class_num = None
            elif args.c2t:
                class_num = num2topic[one_test_dict['topic']]
                keywords = text_topics[p_idx] if args.beta_scale else None
            elif args.n_obj == 0:
                keywords = None
                class_num = None
            else:
                keywords = None
                class_num = None

            image_full_path = args.test_image_prefix_path + '/' + one_test_dict['image_name']

            if args.clip_vis_pkl is not None and os.path.exists(args.clip_vis_pkl):
                image_instance = clip_vis_dict[one_test_dict['image_name']]
            else:
                image_instance = Image.open(image_full_path)

            prompt_text = f"{sos_token} {one_res_dict['title']} {eos_token}" \
                if one_res_dict['title'] != 'none' else sos_token
            decoder_text = "<pad>"

            input_ids = get_prompt_id(prompt_text, generation_model.tokenizer)
            decoder_input_ids = get_prompt_id(decoder_text, generation_model.tokenizer)
            if cuda_available:
                input_ids = input_ids.cuda(device)
                decoder_input_ids = decoder_input_ids.cuda(device)

            # try:
            output_text, ex_time = generation_model.magic_search(input_ids, decoder_input_ids, args.k, args.eta, args.decoding_len,
                                                                 args.alpha, image_instance, clip, clip_text_max_len,
                                                                 condition=args.condition_method, beta=args.beta,
                                                                 keywords=keywords, k2t=(args.k2t or args.obj4topic),
                                                                 enc_dict=enc_dict, mode=mode, guarantee=args.guarantee,
                                                                 only_max=args.only_max, c2t=args.c2t,
                                                                 embed_vis=True if clip_vis_dict is None else False,
                                                                 alpha_scale=args.alpha_scale, beta_scale=args.beta_scale,
                                                                 alpha_activesize=args.alpha_activesize,
                                                                 beta_activesize=args.beta_activesize,
                                                                 topic_classifier=topic_classifier,
                                                                 alpha_upper=args.alpha_upper, beta_upper=args.beta_upper,
                                                                 vis_window_len=args.vis_window_len, kw_mode=args.kw_mode,
                                                                 alpha_dw_mode=args.alpha_dw_mode, class_num=class_num,
                                                                 BOW_top_n=args.BOW_top_n,
                                                                 update_keywords=args.update_keywords,
                                                                 step_gap=args.step_gap, obj4topic=args.obj4topic)

            one_res_dict['prediction'] = output_text
            one_res_dict['time'] = ex_time
            result_list.append(one_res_dict)
            time_list.append(ex_time)
    avg_time = np.mean(time_list)
    print(f'Inference completed! Decoding time is {avg_time}s..')
    with open(full_save_path, 'w', encoding='utf8') as outfile:
        json.dump(result_list, outfile, indent=4, ensure_ascii=False)

    cocoEval = COCOEvalCap(full_save_path)
    cocoEval.evaluate()
    output = {}
    for metric, score in cocoEval.eval.items():
        output[metric] = score
    with open(os.path.join(RESULTS_PREFIX_PATH, save_name + "_eval.json"), 'w') as outfile:
        json.dump(output, outfile, indent=4)


if __name__ == '__main__':
    args = parser.parse_args()
    # args = parser.parse_args('--eta 0.3 --alpha 0.5 --k 50 --condition_method add --decoding_len 256 '
    #                          f'--task flickr30k --c2t --language_model_name {os.path.join(CACHE_DIR, "flan-t5-large")} --seq2seq --alpha_scale --alpha_activesize 0.2 --alpha_upper 0.5'.split())

    original_language_model_name = args.language_model_name
    if args.task in ["humor_flickr", "romantic_flickr"]:
        dataprefix = os.path.join(DATA_DIR, "FlickrStyle")
        args.decoding_len = 25
        if args.task == "humor_flickr":
            args.language_model_name = "PahaII/ZeroGen-flickr10k-humor"
            args.test_path = os.path.join(dataprefix, "humor/humor_caption_test.json")
        elif args.task == "romantic_flickr":
            args.language_model_name = "PahaII/ZeroGen-flickr10k-romantic"
            args.test_path = os.path.join(dataprefix, "romantic/romantic_caption_test.json")
        else:
            raise NotImplementedError

        args.clip_vis_pkl = None
        args.test_image_prefix_path = os.path.join(dataprefix, "test_images")
        if args.k2t:
            args.test_obj_path = os.path.join(dataprefix, "test_image_obj.json")
            args.ed_path = os.path.join(dataprefix, "glove_data/dict_styleflickr10k_obj_glove.pkl")
        if args.seq2seq: args.language_model_name = original_language_model_name
        zerogen_generate(args)

    elif args.task in ["mscoco", "flickr30k"]:
        if args.task=="mscoco":
            dataprefix = os.path.join(DATA_DIR, "mscoco")
            args.test_path = os.path.join(dataprefix, "mscoco_test.json")
            args.train_path = os.path.join(dataprefix, "mscoco_train.json")
            args.test_image_prefix_path = os.path.join(dataprefix, "val2014")
        else:
            dataprefix = os.path.join(DATA_DIR, "flickr30k")
            args.test_path = os.path.join(dataprefix, "flickr30k_test.json")
            args.train_path = os.path.join(dataprefix, "flickr30k_train.json")
            args.test_image_prefix_path = os.path.join(dataprefix, "test_images")

        args.clip_vis_pkl = os.path.join(dataprefix, "dict_clip_vis.pkl")
        args.test_obj_path = os.path.join(dataprefix, "test_image_obj.json")
        args.ed_path = os.path.join(dataprefix, "dict_glove.pkl")
        if args.seq2seq: args.language_model_name = original_language_model_name
        zerogen_generate(args)

    elif args.task == "visnews":
        dataprefix = os.path.join(DATA_DIR, "visnews")
        args.decoding_len = 64
        args.language_model_name = "PahaII/ZeroGen-visnews"
        args.test_path = os.path.join(dataprefix, "simctg_test_news.json")
        args.clip_vis_pkl = None
        args.test_image_prefix_path = dataprefix

        if args.k2t:
            if args.news_label == "pos":
                args.test_obj_path = os.path.join(dataprefix, "test_image_obj_positive.json")
                args.ed_path = os.path.join(dataprefix, "glove_data/dict_visnews_positive_glove.pkl")
            elif args.news_label == "neg":
                args.test_obj_path = os.path.join(dataprefix, "test_image_obj_negative_s.json")
                args.ed_path = os.path.join(dataprefix, "glove_data/dict_visnews_negative_glove.pkl")
            else:
                raise NotImplementedError
        if args.obj4topic:
            args.obj_test_obj_path = os.path.join(dataprefix, "test_image_obj.json")
            args.obj_ed_path = os.path.join(dataprefix, "glove_data/dict_visnews_obj_glove.pkl")
        if args.seq2seq: args.language_model_name = original_language_model_name
        zerogen_generate(args)
