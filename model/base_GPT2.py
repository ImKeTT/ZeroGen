#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
@file: base_GPT2.py
@author: ImKe at 2022/7/27
@email: tuisaac163@gmail.com
@feature: #Enter features here
"""
import torch
import torch.nn as nn
import os, sys, datetime
from transformers import (
    GPT2Tokenizer,
    GPT2Model,
    GPT2LMHeadModel
)
from model.model_utils import (
    PlugAndPlayContrastiveDecodingOneStepFast,
    PlugAndPlayMagicDecodingOneStepFast,
    PlugAndPlayTVDecodingOneStepFast
)


class GPT2LMHeadModelMultimodal(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()

    @torch.no_grad()
    def magic_search_contrastive(self, input_ids, beam_width, alpha, decoding_len, beta, image_instance, clip,
                                 clip_text_max_len, tokenizer):  # , add_token_level_score=False):
        prefix_len = input_ids.size()[1]
        past_key_values, last_hidden_states, logits = None, None, None
        generated = [item for item in input_ids.tolist()]
        input_ids_for_class = input_ids.clone()

        image_embeds = clip.compute_image_representation_from_image_instance(image_instance)

        start_time = datetime.datetime.now()

        # the maximum supported length of generation for SimCTG is 256
        # to support longer generated length, you can re-train the SimCTG model with longer sequences
        decoding_len = decoding_len - prefix_len
        for step in range(decoding_len):
            input_ids, past_key_values, last_hidden_states, logits, input_ids_for_class = \
                PlugAndPlayContrastiveDecodingOneStepFast(
                    self,
                    input_ids,
                    prefix_len,
                    beam_width,
                    alpha,
                    beta,
                    tokenizer,
                    image_embeds,
                    clip,
                    clip_text_max_len,
                    past_key_values,
                    last_hidden_states,
                    logits,
                    first_step=step == 0,
                    input_ids_for_class=input_ids_for_class,
                )
        end_time = datetime.datetime.now()
        time_diff = (end_time - start_time)
        execution_time = time_diff.total_seconds() * 1000
        return self.parse_output_token_list(input_ids_for_class[0], tokenizer)

    def parse_output_token_list(self, output, tokenizer):
        output = output.tolist()
        res_list = []
        for token_id in output:
            res_list.append(token_id)
        text = tokenizer.decode(res_list).strip()
        return ' '.join(text.split()).strip()

    @torch.no_grad()
    def magic_search(self, input_ids, beam_width, alpha, decoding_len, beta, image_instance, clip,
                     clip_text_max_len, tokenizer, condition_method="add", keywords_list=None,
                     keywords_embeds=None, do_sample=False, use_img_feat=False, n_word=1):
        prefix_len = input_ids.size(1)
        past_key_values, last_hidden_states, logits = None, None, None
        generated = [item for item in input_ids.tolist()]
        input_ids_for_class = input_ids.clone()

        if not use_img_feat:
            image_embeds = clip.compute_image_representation_from_image_instance(image_instance)
        else:
            image_embeds = image_instance

        start_time = datetime.datetime.now()

        # the maximum supported length of generation for SimCTG is 256
        # to support longer generated length, you can re-train the SimCTG model with longer sequences
        decoding_len = decoding_len - prefix_len
        for step in range(decoding_len):
            input_ids, past_key_values, last_hidden_states, logits, input_ids_for_class = \
                PlugAndPlayTVDecodingOneStepFast(
                    self,
                    input_ids,
                    prefix_len,
                    beam_width,
                    alpha, beta,
                    tokenizer,
                    image_embeds,
                    clip, clip_text_max_len,
                    past_key_values,
                    last_hidden_states,
                    logits,
                    condition_method=condition_method,
                    keywords_embeds=keywords_embeds,
                    keywords_list=keywords_list,
                    do_sample=do_sample,
                    first_step=(step==0),
                    input_ids_for_class=input_ids_for_class,
                )
        end_time = datetime.datetime.now()
        time_diff = (end_time - start_time)
        execution_time = time_diff.total_seconds() * 1000
        return self.parse_output_token_list(input_ids_for_class[0], tokenizer)
