#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
Taken from the Magic ()
"""

import torch, json
import requests
from torch import nn
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from transformers import CLIPTokenizer
config = json.load(open("./config.json"))
DATA_DIR = config['DATA_DIR']
CACHE_DIR = config['CACHE_DIR']

class CLIP(nn.Module):
    def __init__(self, model_name, device=None):
        super(CLIP, self).__init__()
        # model name: e.g. openai/clip-vit-base-patch32
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name, cache_dir=CACHE_DIR)
        self.model.to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(model_name, cache_dir=CACHE_DIR)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
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

    @torch.no_grad()
    def compute_image_representation_from_image_path(self, image_path):
        if not self.cuda_has_been_checked:
            self.check_cuda()
            self.cuda_has_been_checked = True
        else:
            pass
        # image_path: the path of the image
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values']
        if self.cuda_available:
            pixel_values = pixel_values.cuda(self.device)
        visual_outputs = self.model.vision_model(pixel_values=pixel_values)
        image_embeds = visual_outputs[1]
        image_embeds = self.model.visual_projection(image_embeds) # [1 x embed_dim]
        return image_embeds

    def compute_image_representation_from_image_instance(self, image):
        if not self.cuda_has_been_checked:
            self.check_cuda()
            self.cuda_has_been_checked = True
        else:
            pass
        # image_path: the path of the image
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values']
        if self.cuda_available:
            pixel_values = pixel_values.cuda(self.device)
        visual_outputs = self.model.vision_model(pixel_values=pixel_values)
        image_embeds = visual_outputs[1]
        image_embeds = self.model.visual_projection(image_embeds) # [1 x embed_dim]
        return image_embeds

    def compute_text_representation(self, text_list):
        if not self.cuda_has_been_checked:
            self.check_cuda()
            self.cuda_has_been_checked = True
        else:
            pass
        # text_list: a list of text
        text_inputs = self.tokenizer(text_list, padding=True, return_tensors="pt",
            max_length=self.tokenizer.max_len_single_sentence + 2, truncation=True)
        # self.tokenizer.max_len_single_sentence + 2 = 77
        input_ids, attention_mask = text_inputs['input_ids'], text_inputs['attention_mask']
        if self.cuda_available:
            input_ids = input_ids.cuda(self.device)
            attention_mask = attention_mask.cuda(self.device)
        text_outputs = self.model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_embeds = text_outputs[1]
        text_embeds = self.model.text_projection(text_embeds)
        return text_embeds

    def compute_image_text_similarity_via_embeddings(self, image_embeds, text_embeds, as_logits=True):
        '''
            image_embeds: 1 x embed_dim
            text_embeds: len(text_list) x embed_dim
        '''
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        if as_logits:
            logit_scale = self.model.logit_scale.exp()
            logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        else:
            logits_per_text = torch.matmul(text_embeds, image_embeds.t())
        logits_per_image = logits_per_text.T
        return logits_per_image.softmax(dim=1) # 1 x len(text_list)

    def compute_text_text_similarity_via_embeddings(self, text0_embeds, text1_embeds):
        '''
            image_embeds: 1 x embed_dim
            text_embeds: len(text_list) x embed_dim
        '''
        text0_embeds = text0_embeds / text0_embeds.norm(dim=-1, keepdim=True)
        text1_embeds = text1_embeds / text1_embeds.norm(dim=-1, keepdim=True)
        logit_scale = self.model.logit_scale.exp()
        logits_per_text = torch.matmul(text0_embeds, text1_embeds.t()) * logit_scale
        logits_per_text0 = logits_per_text.T
        return logits_per_text0 # 1 x len(text_list)

    def compute_image_text_similarity_via_raw_text(self, image_embeds, text_list):
        image_embeds = image_embeds.to(self.device)
        text_embeds = self.compute_text_representation(text_list).to(self.device)
        return self.compute_image_text_similarity_via_embeddings(image_embeds, text_embeds)

    def compute_text_text_similarity_via_raw_text(self, text0_list, text1_list, text0_embeds=None):
        if text0_embeds is None:
            text0_embeds = self.compute_text_representation(text0_list)
        text0_embeds = text0_embeds.to(self.device)
        text1_embeds = self.compute_text_representation(text1_list).to(self.device)
        return self.compute_image_text_similarity_via_embeddings(text0_embeds, text1_embeds).mean(0)

    def compute_image_text_similarity_via_logits_transformer(self, image_embeds, text_list, img_transformer, text_transformer):
        text_embeds = self.compute_text_representation(text_list)
        img_logits = img_transformer(image_embeds)
        text_logits = text_transformer(text_embeds)
        return self.compute_image_text_similarity_via_embeddings(img_logits, text_logits, as_logits=False), img_logits, text_logits

    ### -------------------- functions for building index ---------------------- ###
    def compute_batch_index_image_features(self, image_list):
        '''
            # list of image instances
        '''
        if not self.cuda_has_been_checked:
            self.check_cuda()
            self.cuda_has_been_checked = True
        else:
            pass
        # image_path: the path of the image
        inputs = self.processor(images=image_list, return_tensors="pt")
        pixel_values = inputs['pixel_values']
        if self.cuda_available:
            pixel_values = pixel_values.cuda(self.device)
        visual_outputs = self.model.vision_model(pixel_values=pixel_values)
        image_embeds = visual_outputs[1]
        image_embeds = self.model.visual_projection(image_embeds) # [1 x embed_dim]
        return image_embeds # len(image_list) x embed_dim

    def compute_batch_index_text_representation(self, text_list):
        if not self.cuda_has_been_checked:
            self.check_cuda()
            self.cuda_has_been_checked = True
        else:
            pass
        # text_list: a list of text
        #text_inputs = self.tokenizer(text_list, padding=True, return_tensors="pt")
        text_inputs = self.tokenizer(text_list, padding=True, return_tensors="pt",
            max_length=self.tokenizer.max_len_single_sentence + 2, truncation=True)
        input_ids, attention_mask = text_inputs['input_ids'], text_inputs['attention_mask']
        if self.cuda_available:
            input_ids = input_ids.cuda(self.device)
            attention_mask = attention_mask.cuda(self.device)
        text_outputs = self.model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_embeds = text_outputs[1]
        text_embeds = self.model.text_projection(text_embeds)
        return text_embeds
        #logit_scale = self.model.logit_scale.exp()
        #text_embeds = text_embeds * logit_scale
        #return text_embeds

