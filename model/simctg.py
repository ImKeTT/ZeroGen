import os
import sys
import operator
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
porter = PorterStemmer()

CACHE_DIR='/mnt/data0/tuhq21/.cache/torch/transformers'

train_fct = CrossEntropyLoss()
val_fct = CrossEntropyLoss(reduction='none')


class SimCTG(nn.Module):
    def __init__(self, model_name, sos_token, pad_token, converter_table=None):
        super(SimCTG, self).__init__()
        from transformers import AutoTokenizer, GPT2LMHeadModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
        self.sos_token, self.sos_token_id = self.add_special_token(sos_token)
        print('sos token is {}, sos token id is {}'.format(self.sos_token, self.sos_token_id))
        self.pad_token, self.pad_token_id = self.add_special_token(pad_token)
        print('pad token is {}, pad token id is {}'.format(self.pad_token, self.pad_token_id))
        self.eos_token, self.eos_token_id = self.tokenizer.bos_token, self.tokenizer.bos_token_id
        print('eos token is {}, eos token id is {}'.format(self.eos_token, self.eos_token_id))
        self.model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=CACHE_DIR)
        self.vocab_size = len(self.tokenizer)
        print('Resizing model embedding...')
        self.model.resize_token_embeddings(len(self.tokenizer))
        print('Model embedding resized!')
        self.embed_dim = self.model.config.hidden_size

        self.converter_table=converter_table
        if self.converter_table is not None:
            self.reshape_converter_table()

    def reshape_converter_table(self):
        converter_size, converter_mid_size = self.converter_table.shape[0], self.converter_table.shape[1]
        if converter_size < len(self.tokenizer):
            added_len = len(self.tokenizer) - converter_size
            self.converter_table = np.concatenate([self.converter_table, np.zeros((added_len,
                                                                                   converter_mid_size))], axis=0)
            print(f"Adding {added_len} tokens for converter_table")

    def add_special_token(self, special_token):
        if special_token in self.tokenizer.vocab:
            print(special_token + ' token exists.')
        else:
            print('Add token to the tokenizer.')
            print('Original vocabulary size is {}'.format(len(self.tokenizer)))
            self.tokenizer.add_tokens([special_token])
            print('Vocabulary size after extension is {}'.format(len(self.tokenizer)))
            assert len(self.tokenizer.convert_tokens_to_ids([special_token])) == 1
        special_token_id = self.tokenizer.convert_tokens_to_ids([special_token])[0]
        return special_token, special_token_id

    def compute_logits_and_hidden_states(self, input_ids):
        # used for advanced decoding
        # input_ids: 1 x seqlen
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        logits = outputs.logits
        return last_hidden_states, logits

    def forward(self, input_ids, labels, margin):
        bsz, seqlen = input_ids.size()
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        logits = outputs.logits
        assert logits.size() == torch.Size([bsz, seqlen, self.vocab_size])
        last_hidden_states = outputs.hidden_states[-1]
        assert last_hidden_states.size() == torch.Size([bsz, seqlen, self.embed_dim])
        mle_loss = train_fct(logits.view(-1, self.vocab_size), labels.view(-1))

        norm_rep = last_hidden_states / last_hidden_states.norm(dim=2, keepdim=True)
        cosine_scores = torch.matmul(norm_rep, norm_rep.transpose(1, 2))
        assert cosine_scores.size() == torch.Size([bsz, seqlen, seqlen])
        cl_loss = contrastive_loss(margin, cosine_scores, input_ids, self.pad_token_id, prefix_len=0)
        return mle_loss, cl_loss

    def eval_loss(self, input_ids, labels):
        bsz, seqlen = input_ids.size()
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        logits = outputs.logits
        assert logits.size() == torch.Size([bsz, seqlen, self.vocab_size])
        last_hidden_states = outputs.hidden_states[-1]
        assert last_hidden_states.size() == torch.Size([bsz, seqlen, self.embed_dim])
        mle_loss = val_fct(logits.view(-1, self.vocab_size), labels.view(-1))
        assert mle_loss.size() == torch.Size([bsz * seqlen])
        mask_tmp = labels.masked_fill(~labels.eq(-100), 1.0)
        mask = mask_tmp.masked_fill(mask_tmp.eq(-100), 0.0)
        # sum
        mle_loss_sum = torch.sum(mle_loss)
        token_num_sum = torch.sum(mask)
        return mle_loss_sum, token_num_sum

    def save_model(self, ckpt_save_path):
        import os
        if os.path.exists(ckpt_save_path):
            pass
        else:  # recursively construct directory
            os.makedirs(ckpt_save_path, exist_ok=True)
        # save model
        self.model.save_pretrained(ckpt_save_path)
        # save tokenizer
        self.tokenizer.save_pretrained(ckpt_save_path)

    def parse_sentences(self, text, num_of_sentences_to_keep):
        item_list = text.split('.')
        res_list = item_list[:num_of_sentences_to_keep]
        if len(item_list) > num_of_sentences_to_keep:
            res_text = '.'.join(res_list).strip('.') + '.'
        else:
            res_text = '.'.join(res_list).strip('.').strip()
        return res_text

    def parse_generated_result(self, output, num_of_sentences_to_keep):
        output_text = self.tokenizer.decode(output)
        item_list = output_text.split(self.eos_token)
        full_text = self.eos_token.join(item_list[:2]).strip()
        full_text = self.parse_sentences(full_text, num_of_sentences_to_keep)
        generated_text = item_list[1].strip()
        generated_text = self.parse_sentences(generated_text, num_of_sentences_to_keep)
        return full_text, generated_text

    # decoding functions
    # ------------------------------------------------------- #

    def parse_output_token_list(self, output):
        output = output.tolist()
        res_list = []
        for token_id in output:
            if token_id == self.sos_token_id:
                continue
            elif token_id == self.eos_token_id:
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
            pred_word = predicted_text.split()[-1]#.split('<|endoftext|>')[-1]

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
    def magic_search(self, input_ids, beam_width, alpha, decoding_len, beta, image_instance, clip,
                     clip_text_max_len, condition="add", eta=0., k2t=False, c2t=False, keywords=None,
                     class_num=None, enc_dict={}, mode="next", guarantee=False, topic_classifier=None,
                     T_time=1, only_max=False, embed_vis=True, beta_scale=False, eta_scale=False,
                     beta_activesize=None, eta_activesize=None, beta_upper=10.0, eta_upper=10.0,
                     vis_window_len=1, beta_dw_mode=1, kw_mode='sum', BOW_top_n=None,
                     update_keywords=False, step_gap=1, alpha_scale=False, obj4topic=False):  # , add_token_level_score=False):

        prefix_len = input_ids.size()[1]
        past_key_values, last_hidden_states, logits = None, None, None
        generated = [item for item in input_ids.tolist()]
        input_ids_for_class = input_ids.clone()

        if embed_vis:
            image_embeds = clip.compute_image_representation_from_image_instance(image_instance)
        else:
            image_embeds = torch.tensor(image_instance).cuda()

        start_time = datetime.datetime.now()

        keywords_sim, keywords_gpt, keywords_gpt_reverse, keywords_sims, obj_keywords_sims = None, None, None, None, None
        if k2t and keywords is not None and len(keywords):
            assert c2t is False
            keywords_enc, keywords_gpt = self.get_keywords(keywords, enc_dict, mode)
            # keywords_sim = self.get_keyword_sim(keywords_enc, keywords_gpt, self.converter_table, guarantee,
            #                                     kw_mode, only_max, obj4topic)
            keywords_sims, obj_keywords_sims = self.get_keyword_sims(keywords_enc, keywords_gpt, self.converter_table, guarantee,only_max, obj4topic)

        if beta_dw_mode==2:
            beta = torch.full([1, beam_width], beta, device=self.model.device)
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

            input_ids, past_key_values, last_hidden_states, logits, input_ids_for_class, cur_logits, vis_window_sim = \
                PlugAndPlayContrastiveDecodingOneStepFast(
                    self.model,
                    input_ids,
                    prefix_len,
                    beam_width,
                    alpha,
                    beta,
                    eta,
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
                    beta_scale=beta_scale,
                    vis_window_len=vis_window_len,
                    topic_classifier=topic_classifier,
                    c2t=c2t,
                    num_topic=class_num ## topic number
                )
            if eta_scale:
                assert eta_activesize is not None
                alter_scale = self.exam_BOW_distribution(keywords, torch.softmax(cur_logits.mean(0), dim=-1), BOW_top_n) / eta_activesize
                eta = eta * alter_scale
                eta = max(eta, eta_upper)

            if beta_scale and vis_window_sim is not None:
                assert beta_activesize is not None
                if beta_dw_mode==1:
                    ## vis_window_sim [1, beam_width]
                    alter_scale = vis_window_sim.mean(-1) / beta_activesize
                    beta = beta * alter_scale
                    beta = max(beta, beta_upper)
                elif beta_dw_mode==2:
                    alter_scale = vis_window_sim / beta_activesize
                    beta = beta * alter_scale
                    beta = torch.minimum(beta, torch.tensor(beta_upper, device=beta.device))
                    beta = beta.view([beam_width])
                else:
                    raise NotImplementedError
            if alpha_scale:
                alpha = max(alpha - int(step / 5) * 0.001, 0.25)

            if k2t and update_keywords and (step % step_gap==0) and keywords is not None and len(keywords):
                cur_text = self.tokenizer.decode(input_ids_for_class[0][prefix_len:]).strip()
                cur_text = ' '.join(cur_text.split()).strip()
                keywords = self.update_keywords(cur_text, keywords)

                keywords_enc, keywords_gpt = self.get_keywords(keywords, enc_dict, mode)
                keywords_sim = self.get_keyword_sim(keywords_enc, keywords_gpt, self.converter_table, guarantee,
                                                    kw_mode, only_max)


            # if k2t and keywords is not None and len(keywords):
            #     pred_word, _, _ = self.get_prediction(input_ids_for_class[0][:-1], keywords_gpt, input_ids[0], guarantee, T_time, time)
            #     # Update counters if word was predicted
            #     pred_word_stem = porter.stem(pred_word.lower())
            #     # guide_next = guide
            #     time_next = time + 1
            #     T_time_next = T_time
            #     if pred_word_stem in guide_word_stems:
            #         ind = guide_word_stems.index(pred_word_stem)
            #         keywords = keywords[:ind] + keywords[ind + 1:]
            #         # guide_probs = guide_probs + [(pred_word_stem, proba[predicted_index].item())]
            #         guide_next = False
            #         time_next = 1
            #         T_time_next = T_time - time + 1
            #
            #     eta = get_weight(eta, guarantee, T_time_next, time_next)

        end_time = datetime.datetime.now()
        time_diff = (end_time - start_time)
        execution_time = time_diff.total_seconds() * 1000

        return self.parse_output_token_list(input_ids_for_class[0][prefix_len:]), execution_time

    def fast_contrastive_search(self, input_ids, beam_width, alpha, decoding_len):
        '''
           input_ids: prefix input; 1 x prefix_len
           decoding_len: how many tokens to generate
           beam_width: size of candidate pool during decoding
           alpha: regulates importance of model confidence and degeneration penalty
        '''
        self.model.eval()
        from model_utils import ContrastiveDecodingOneStepFast
        # sanity check
        assert alpha >= 0. and alpha <= 1.0

        # fast mode
        prefix_len = input_ids.size()[1]
        batch_size, seqlen = input_ids.size()
        # generated = [[] for _ in range(batch_size)]
        generated = [item for item in input_ids.tolist()]
        past_key_values = None
        last_hidden_states = None
        logits = None
        decoding_len = decoding_len - prefix_len
        for step in range(decoding_len):
            input_ids, past_key_values, last_hidden_states, logits = ContrastiveDecodingOneStepFast(
                self.model,
                input_ids,
                beam_width,
                alpha,
                past_key_values,
                last_hidden_states,
                self.tokenizer,
                logits,
                first_step=step == 0,
            )
            tokens = input_ids.squeeze(dim=-1).tolist()
            for idx, t in enumerate(tokens):
                generated[idx].append(t)
        return self.parse_output_token_list(torch.LongTensor(generated[0]))

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