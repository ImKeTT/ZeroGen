import os
import sys
from tqdm import tqdm
from operator import itemgetter
import torch
from torch import nn
import random
import argparse
import numpy as np
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
sys.path.append("../")
from model.model_utils import contrastive_loss
from model.model_utils import PlugAndPlayContrastiveDecodingOneStepFast
from model.model_utils import get_weight
import datetime
from nltk.stem import PorterStemmer, LancasterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import json
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
porter = PorterStemmer()

config = json.load(open("./config.json"))
DATA_DIR = config['DATA_DIR']
CACHE_DIR = config['CACHE_DIR']

train_fct = CrossEntropyLoss()
val_fct = CrossEntropyLoss(reduction='none')


class ZeroGen(nn.Module):
    def __init__(self, model_name, seq2seq=True, converter_table=None):
        super(ZeroGen, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
        if seq2seq:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=CACHE_DIR)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=CACHE_DIR)
        self.vocab_size = len(self.tokenizer)
        self.embed_dim = self.model.config.hidden_size
        self.seq2seq=seq2seq

        self.converter_table = converter_table
        if self.converter_table is not None:
            self.reshape_converter_table()

    def reshape_converter_table(self):
        converter_size, converter_mid_size = self.converter_table.shape[0], self.converter_table.shape[1]
        if converter_size < len(self.tokenizer):
            added_len = len(self.tokenizer) - converter_size
            self.converter_table = np.concatenate([self.converter_table, np.zeros((added_len, converter_mid_size))], axis=0)
            print(f"Adding {added_len} tokens for converter_table")

    # decoding functions
    # ------------------------------------------------------- #

    def parse_output_token_list(self, output):
        output = output.tolist()
        res_list = []
        for token_id in output:
            # if token_id == self.tokenizer.sos_token_id:
            #     continue
            if token_id == self.tokenizer.eos_token_id:
                break
            else:
                res_list.append(token_id)
        text = self.tokenizer.decode(res_list).strip()
        return ' '.join(text.split()).strip()

    def get_keywords(self, keywords, enc_dict, mode):
        keywords_ = [w for w in keywords]

        # Select the next guide word(s)
        if keywords_:
            if mode == 'next':
                keywords_ = [keywords_[0]]
            elif mode == 'random':
                keywords_ = [random.choice(keywords_)]

        keywords_enc = [enc_dict[w] for w in keywords_]
        keywords_gpt = {self.tokenizer.encode(" "+w)[0]: w for w in keywords_}

        return keywords_enc, keywords_gpt

    def get_prediction(self, indexed_tokens, keywords_gpt, predicted_index, guarantee, T_time, time):

        if guarantee and time > T_time:
            predicted_index = list(keywords_gpt.keys())[0]
        if guarantee and predicted_index in keywords_gpt:
            predicted_text = self.tokenizer.batch_decode(indexed_tokens)[0] + ' ' + keywords_gpt[predicted_index]
            # this_sequence = self.tokenizer.decode(indexed_this_seq) + ' ' + keywords_gpt[predicted_index]
            pred_word = keywords_gpt[predicted_index]
        else:
            predicted_text = self.tokenizer.batch_decode(torch.cat([indexed_tokens, predicted_index]))[0]
            # this_sequence = self.tokenizer.decode(indexed_this_seq + [predicted_index])
            pred_word = predicted_text.split()[-1] #.split('<|endoftext|>')[-1]

        return pred_word, predicted_text, predicted_index

    def get_keyword_sim(self, keywords_enc, keywords_gpt, converter_table, guarantee, mode, only_max, obj4topic):
        if len(keywords_enc) > 1:
            sims = np.array([cosine_similarity(np.reshape(w, (1, -1)), converter_table) for w in keywords_enc])
            if guarantee:
                for i, w in enumerate(keywords_gpt):
                    sims[i][0][w] = 1
            if obj4topic:
                sim_obj = sims[0]
                sims = sims[1:]
            if mode == 'max':
                sim = np.max(sims, axis=0)
            elif mode == 'sum':
                sim = np.sum(sims, axis=0)
            elif mode == "mean":
                sim = np.mean(sims, axis=0)
            else:
                raise Exception("keywords_enc length is greater than 1 so expect to be in mode 'max' or 'all'")
            if obj4topic:
                sim = np.mean(sim_obj, sim)
        else:
            sim = cosine_similarity(np.reshape(keywords_enc[0], (1, -1)), converter_table)

        # Only the target word, not the neighbour (as measured by cosine similarity)
        if only_max == True:
            sim_aux = np.zeros_like(sim)
            sim_aux[0, sim.argmax()] = sim.max()
            sim = np.squeeze(sim_aux)
        else:
            sim = np.clip(np.squeeze(sim), a_min=0, a_max=None)

        return sim

    def get_keyword_sims(self, keywords_enc, keywords_gpt, converter_table, guarantee, only_max, obj4topic):
        sim_obj = None
        if len(keywords_enc) > 1:
            sims = np.array([cosine_similarity(np.reshape(w, (1, -1)), converter_table) for w in keywords_enc])
            if guarantee:
                for i, w in enumerate(keywords_gpt):
                    sims[i][0][w] = 1
            if obj4topic:
                sim_obj = sims[0][0]
                sims = sims[1:]
        else:
            sims = cosine_similarity(np.reshape(keywords_enc[0], (1, -1)), converter_table)

        # Only the target word, not the neighbour (as measured by cosine similarity)
        if only_max == True:
            sim_aux = np.zeros_like(sims)
            sim_aux[0, sims.argmax()] = sims.max()
            sims = np.squeeze(sim_aux)
        else:
            sims = np.clip(np.squeeze(sims), a_min=0, a_max=None)

        return sims, sim_obj

    def exam_BOW_distribution(self, keywords, probs, BOW_top_n=None):
        # good_index = [[input_id1], [inputid2], ...]
        # print(probs)
        good_index = []
        for line in keywords:
            good_index.append(self.tokenizer(' '+line.strip()).input_ids)
        sum = []
        for indices in good_index:
            tmp_probs = 0
            for ids in indices:
                tmp_probs += probs[ids].item()
            sum.append(tmp_probs)

        if BOW_top_n is None:
            sum = np.sum(sum)
        else:
            sum.sort(reverse=True)
            sum = np.sum(sum[:BOW_top_n])
        return sum

    def update_keywords(self, generated_phrase, keywords):
        keywords_updated = []
        for i, keyword in enumerate(keywords):
            if keyword not in generated_phrase:
                keywords_updated.append(keyword)
        return keywords_updated

    @torch.no_grad()
    def magic_search(self, input_ids, decoder_input_ids, beam_width, eta, decoding_len, alpha, image_instance, clip,
                     clip_text_max_len, condition="add", beta=0., k2t=False, c2t=False, keywords=None,
                     class_num=None, enc_dict={}, mode="next", guarantee=False, topic_classifier=None,
                     only_max=False, embed_vis=True, alpha_scale=False, beta_scale=False,
                     alpha_activesize=None, beta_activesize=None, alpha_upper=10.0, beta_upper=10.0,
                     vis_window_len=1, alpha_dw_mode=1, kw_mode='sum', BOW_top_n=None,
                     update_keywords=False, step_gap=1, obj4topic=False):  # , add_token_level_score=False):

        prefix_len = input_ids.size()[1] if not self.seq2seq else decoder_input_ids.size()[1]
        past_key_values, last_hidden_states, logits = None, None, None
        generated = [item for item in input_ids.tolist()]
        input_ids_for_class = input_ids.clone() if not self.seq2seq else decoder_input_ids.clone()

        if embed_vis:
            image_embeds = clip.compute_image_representation_from_image_instance(image_instance)
        else:
            image_embeds = torch.tensor(image_instance).cuda()

        start_time = datetime.datetime.now()

        keywords_sim, keywords_gpt, keywords_gpt_reverse, keywords_sims, obj_keywords_sims = None, None, None, None, None
        if k2t and keywords is not None and len(keywords):
            assert c2t is False
            keywords_enc, keywords_gpt = self.get_keywords(keywords, enc_dict, mode)
            keywords_sims, obj_keywords_sims = self.get_keyword_sims(keywords_enc, keywords_gpt, self.converter_table, guarantee,only_max, obj4topic)

        if alpha_dw_mode==2:
            alpha = torch.full([1, beam_width], alpha, device=self.model.device)
        # the maximum supported length of generation for SimCTG is 256
        # to support longer generated length, you can re-train the SimCTG model with longer sequences
        decoding_len = decoding_len - prefix_len
        for step in range(decoding_len):
            if keywords_sims is not None:
                if kw_mode == "max":
                    keywords_sim = np.max(keywords_sims, axis=0)
                elif kw_mode == "mean":
                    keywords_sim = np.mean(keywords_sims, axis=0)
                elif kw_mode == "sum":
                    keywords_sim = np.sum(keywords_sims, axis=0)
                elif kw_mode == "random":
                    keywords_sim = keywords_sims[np.random.choice(len(keywords_sims))]
                if obj_keywords_sims is not None:
                    obj_keywords_sim = np.stack((obj_keywords_sims, keywords_sim), axis=0)
                    keywords_sim = np.max(obj_keywords_sim, axis=0)

            input_ids, decoder_input_ids, past_key_values, last_hidden_states, logits, input_ids_for_class, cur_logits, vis_window_sim = \
                PlugAndPlayContrastiveDecodingOneStepFast(
                    self.model,
                    input_ids,
                    decoder_input_ids,
                    prefix_len,
                    beam_width,
                    eta,
                    alpha,
                    beta,
                    self.tokenizer,
                    image_embeds,
                    clip,
                    clip_text_max_len,
                    past_key_values,
                    last_hidden_states,
                    logits,
                    first_step=step == 0,
                    input_ids_for_class=input_ids_for_class,
                    condition=condition,
                    k2t=k2t,
                    keywords_sim=keywords_sim,
                    alpha_scale=alpha_scale,
                    vis_window_len=vis_window_len,
                    topic_classifier=topic_classifier,
                    c2t=c2t,
                    num_topic=class_num, ## topic number
                    seq2seq=self.seq2seq,
                )
            if beta_scale:
                assert beta_activesize is not None
                alter_scale = self.exam_BOW_distribution(keywords, torch.softmax(cur_logits.mean(0), dim=-1), BOW_top_n) / beta_activesize
                beta = beta * alter_scale
                beta = max(beta, beta_upper)

            if alpha_scale and vis_window_sim is not None:
                assert alpha_activesize is not None
                if alpha_dw_mode==1:
                    ## vis_window_sim [1, beam_width]
                    alter_scale = vis_window_sim.mean(-1) / alpha_activesize
                    alpha = alpha * alter_scale
                    alpha = max(alpha, alpha_upper)
                elif alpha_dw_mode==2:
                    alter_scale = vis_window_sim / alpha_activesize
                    alpha = alpha * alter_scale
                    alpha = torch.minimum(alpha, torch.tensor(alpha_upper, device=alpha.device))
                    alpha = alpha.view([beam_width])
                else:
                    raise NotImplementedError

            if k2t and update_keywords and (step % step_gap==0) and keywords is not None and len(keywords):
                cur_text = self.tokenizer.decode(input_ids_for_class[0][prefix_len:]).strip()
                cur_text = ' '.join(cur_text.split()).strip()
                keywords = self.update_keywords(cur_text, keywords)

                keywords_enc, keywords_gpt = self.get_keywords(keywords, enc_dict, mode)
                keywords_sim = self.get_keyword_sim(keywords_enc, keywords_gpt, self.converter_table, guarantee,  kw_mode, only_max, obj4topic)

        end_time = datetime.datetime.now()
        time_diff = (end_time - start_time)
        execution_time = time_diff.total_seconds() * 1000

        return self.parse_output_token_list(input_ids_for_class[0][prefix_len:]), execution_time

    def top_k_sampling(self, input_ids, k, decoding_len):
        _, prefix_len = input_ids.size()
        output = self.model.generate(
            input_ids,
            do_sample=True,
            max_length=decoding_len,
            top_p=1.0,
            top_k=k)
        return self.parse_output_token_list(output[0])

    def nucleus_sampling(self, input_ids, nucleus_p, decoding_len):
        _, prefix_len = input_ids.size()
        output = self.model.generate(
            input_ids,
            do_sample=True,
            max_length=decoding_len,
            top_p=nucleus_p,
            top_k=0)
        return self.parse_output_token_list(output[0])