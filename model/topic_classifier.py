#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
Taken from the Magic ()
"""

import torch
import requests
from torch import nn
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSequenceClassification
CACHE_DIR='/mnt/data0/tuhq21/.cache/torch/transformers'

class TopicClassifier(nn.Module):
    def __init__(self, model_name, device=None):
        super(TopicClassifier, self).__init__()
        # model name: e.g. RamAnanth1/distilroberta-base-finetuned-topic-news
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=CACHE_DIR)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
        self.cuda_has_been_checked = False

    def check_cuda(self):
        self.cuda_available = next(self.model.parameters()).is_cuda
        if self.device is None:
            self.device = next(self.model.parameters()).get_device()
        if self.cuda_available and self.device != "cpu":
            print ('Cuda is available.')
            print ('Device is {}'.format(self.device))
        else:
            print ('Cuda is not available.')
            print ('Device is {}'.format(self.device))

    def compute_text_probs(self, text_list, topic_num):
        if not self.cuda_has_been_checked:
            self.check_cuda()
            self.cuda_has_been_checked = True
        else:
            pass
        # text_list: a list of text
        text_inputs = self.tokenizer(text_list, padding=True, return_tensors="pt")
        # self.tokenizer.max_len_single_sentence + 2 = 77
        input_ids, attention_mask = text_inputs['input_ids'], text_inputs['attention_mask']
        if self.cuda_available:
            input_ids = input_ids.cuda(self.device)
            attention_mask = attention_mask.cuda(self.device)
        text_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_probs = text_outputs[0].softmax(dim=1)
        return text_probs[:, topic_num]
    def predict_text_class(self, text_list, topic_num):
        if not self.cuda_has_been_checked:
            self.check_cuda()
            self.cuda_has_been_checked = True
        else:
            pass
        # text_list: a list of text
        text_inputs = self.tokenizer(text_list, padding=True, return_tensors="pt")
        # self.tokenizer.max_len_single_sentence + 2 = 77
        input_ids, attention_mask = text_inputs['input_ids'], text_inputs['attention_mask']
        if self.cuda_available:
            input_ids = input_ids.cuda(self.device)
            attention_mask = attention_mask.cuda(self.device)
        text_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_probs = text_outputs[0].softmax(dim=1)
        return int(torch.argmax(text_probs).item()==topic_num)