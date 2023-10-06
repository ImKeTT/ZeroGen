#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
@file: model_utils.py
@author: ImKe at 2022/7/13
@email: tuisaac163@gmail.com
@feature: #Enter features here
"""
import torch, os
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def enlarge_past_key_values(past_key_values, beam_width):
    # from [B, num_head, seq_len, esz] to [B*K, num_head, seq_len, esz]
    new_key_values = []
    for layer in past_key_values:
        items = []
        for item in layer:
            # item is the key and value matrix
            bsz, num_head, seq_len, esz = item.size()
            item = item.unsqueeze(1).expand(-1, beam_width, -1, -1, -1).reshape(bsz*beam_width, num_head, seq_len, esz)    # [bsz*beam, num_head, seq_len, esz]
            items.append(item)
        new_key_values.append(items)
    return new_key_values

def select_past_key_values(past_key_values, beam_width, selected_idx):
    '''select_idx: [B]'''
    new_key_values = []
    for layer in past_key_values:
        items = []
        for item in layer:
            bsz_and_beam, num_head, seq_len, esz = item.size()
            bsz = int(bsz_and_beam//beam_width)
            item = torch.stack(torch.split(item, beam_width, dim=0))    # [B, K, num_head, seq_len, esz]
            item = item[range(bsz), selected_idx, :, :, :]   # [B, num_head, seq_len, esz]
            items.append(item)
        new_key_values.append(items)
    return new_key_values

# ========== fast plug and play version ========= #
def plug_and_play_fast_ranking(
        context_hidden,
        next_hidden,
        next_top_k_ids,
        next_top_k_probs,
        eta,
        alpha,
        batch_class_score,
        beam_width,
        condition="add",
        beta = 0.,
        topic_batch_score = None
):
    '''
        context_hidden: beam_width x context_len x embed_dim
        next_hidden: beam_width x 1 x embed_dim
        next_top_k_ids: beam_width x 1
        batch_class_score: beam_width x 1
    '''
    _, context_len, embed_dim = context_hidden.size()
    norm_context_hidden = context_hidden / context_hidden.norm(dim=2, keepdim=True)
    norm_next_hidden = next_hidden / next_hidden.norm(dim=2, keepdim=True)
    cosine_matrix = torch.matmul(norm_context_hidden, norm_next_hidden.transpose(1, 2)).squeeze(-1)
    scores, _ = torch.max(cosine_matrix, dim=-1)
    next_top_k_probs = next_top_k_probs.view(-1)
    if condition=="add":
        scores = (1.0 - eta) * next_top_k_probs - eta * scores + \
                 alpha * batch_class_score.view([beam_width])
        if topic_batch_score is not None:
            scores += beta * topic_batch_score.view([beam_width])
    elif condition=="mul":
        next_top_k_probs /= next_top_k_probs.sum(-1)
        batch_class_score /= batch_class_score.sum(-1)
        scores = ((1.0 - eta) * next_top_k_probs - eta * scores) * \
                 pow(batch_class_score.view([beam_width]), -1 * alpha)
    else:
        raise NotImplementedError
    scores = torch.stack(torch.split(scores, beam_width))
    selected_idx = scores.max(dim=-1)[1]
    return selected_idx

def plug_and_play_fast_vis_embeds_bias(
        next_top_k_probs,
        eta,
        alpha,
        batch_class_score,
        beam_width,
        condition_method="add"
):
    '''
        Magic search score without contrastive searching
        context_hidden: beam_width x context_len x embed_dim
        next_hidden: beam_width x 1 x embed_dim
        next_top_k_ids: beam_width x 1
        batch_class_score: beam_width x 1
    '''
    next_top_k_probs = next_top_k_probs.view(-1)
    if condition_method == "add":
        scores = eta * next_top_k_probs + alpha * batch_class_score.view([beam_width])
    elif condition_method == "mul":
        next_top_k_probs /= next_top_k_probs.sum(-1)
        batch_class_score /= batch_class_score.sum(-1)
        scores = next_top_k_probs * pow(batch_class_score.view([beam_width]), -1 * alpha)
    else:
        raise NotImplementedError
    scores = torch.stack(torch.split(scores, beam_width))
    selected_idx = scores.max(dim=-1)[1]
    return selected_idx

def PlugAndPlayMagicDecodingOneStepFast(model, input_ids, prefix_len, beam_width, eta, alpha,
                                              tokenizer, image_embeds, clip, clip_text_max_len, past_key_values,
                                              last_hidden_states,
                                              logit_for_next_step, first_step=False,
                                              input_ids_for_class=None):
    if first_step:
        output = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
        past_key_values = output.past_key_values
        last_hidden_states = output.hidden_states[-1]  # [B, S, E]
        logit_for_next_step = output.logits[:, -1, :]  # [B, V]

    bsz, seqlen, embed_dim = last_hidden_states.size()
    next_probs = F.softmax(logit_for_next_step, dim=-1)
    _, top_k_ids = torch.topk(logit_for_next_step, dim=-1, k=beam_width)
    top_k_probs = torch.gather(next_probs, dim=1, index=top_k_ids)

    # compute the new hidden
    past_key_values = enlarge_past_key_values(past_key_values, beam_width)
    output = model(
        input_ids=top_k_ids.view(-1, 1),
        attention_mask=torch.ones_like(top_k_ids.view(-1, 1)),
        past_key_values=past_key_values,
        output_hidden_states=True,
        use_cache=True,
    )
    past_key_values = output.past_key_values
    logits = output.logits[:, -1, :]
    next_hidden = output.hidden_states[-1]

    # prepare for the classification model
    input_ids_for_class_ = torch.cat([
        input_ids_for_class.unsqueeze(1).expand(-1, beam_width, -1).reshape(bsz * beam_width, seqlen),
        top_k_ids.view(-1, 1)
    ], dim=-1
    )

    batch_text_list = []
    for one_input_id in input_ids_for_class_:
        one_text = tokenizer.decode(one_input_id[prefix_len:][-clip_text_max_len:])
        # we only consider the class score of the generated text continuation
        batch_text_list.append(one_text)
    batch_score = clip.compute_image_text_similarity_via_raw_text(image_embeds, batch_text_list)

    selected_idx = plug_and_play_fast_vis_embeds_bias(
        top_k_probs,
        eta,
        alpha,
        batch_score,
        beam_width
    )
    # prepare for the next step
    next_id = top_k_ids[range(len(top_k_ids)), selected_idx].unsqueeze(-1)
    next_hidden = torch.stack(torch.split(next_hidden.squeeze(dim=1), beam_width))
    next_hidden = next_hidden[range(bsz), selected_idx, :]
    last_hidden_states = torch.cat([last_hidden_states, next_hidden.unsqueeze(1)], dim=1)
    past_key_values = select_past_key_values(past_key_values, beam_width, selected_idx)
    logits = torch.stack(torch.split(logits, beam_width))[range(bsz), selected_idx, :]
    input_ids_for_class = torch.cat([input_ids_for_class, next_id], dim=-1)
    return next_id, past_key_values, last_hidden_states, logits, input_ids_for_class


def get_weight(weight, guarantee, T_time, time):
    if guarantee:
        if T_time == 0:
            T_time = 1
        rate = (1 / T_time) * np.log(100 / weight)  # 100 is the maximum value the weight will reach
        weight = weight * np.exp(rate * time)

    return weight

def PlugAndPlayContrastiveDecodingOneStepFast(model, input_ids, decoder_input_ids, prefix_len, beam_width, eta, alpha, beta,
                                              simctg_tokenizer, image_embeds, clip, clip_text_max_len, past_key_values,
                                              last_hidden_states, logit_for_next_step, first_step=False,
                                              input_ids_for_class=None, condition="add", topic_classifier=None,
                                              c2t=False, num_topic=None,  k2t=False, keywords_sim=None,
                                              alpha_scale=False, vis_window_len=1, seq2seq=True):  # , add_token_level_score=False):
    '''
        model: the generation model, e.g., gpt2
        input_ids: 1 x seqlen
    '''

    if first_step:
        if seq2seq:
            output = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids,
                           past_key_values=past_key_values, use_cache=True,
                           output_hidden_states=True)
        else:
            output = model(input_ids=input_ids, past_key_values=past_key_values,
                           use_cache=True, output_hidden_states=True)
        past_key_values = output.past_key_values
        if seq2seq:
            last_hidden_states = output.decoder_hidden_states[-1]  # [B, S, E]
        else:
            last_hidden_states = output.hidden_states[-1]
        logit_for_next_step = output.logits[:, -1, :]  # [B, V]

    if k2t and keywords_sim is not None:
        assert c2t is False
        ## Get log-softmax of logits for modification
        logit_for_next_step = F.log_softmax(logit_for_next_step, dim=-1)
        logit_for_next_step = logit_for_next_step + torch.tensor(keywords_sim * beta).cuda()

    bsz, seqlen, embed_dim = last_hidden_states.size()
    next_probs = F.softmax(logit_for_next_step, dim=-1)
    _, top_k_ids = torch.topk(logit_for_next_step, dim=-1, k=beam_width)
    top_k_probs = torch.gather(next_probs, dim=1, index=top_k_ids)

    # compute the new hidden
    past_key_values = enlarge_past_key_values(past_key_values, beam_width)
    input_ids_beam_search = input_ids.repeat(beam_width, 1)

    if seq2seq:
        output = model(
            input_ids=input_ids_beam_search,
            decoder_input_ids=top_k_ids.view(-1, 1),
            # attention_mask=torch.ones_like(top_k_ids.view(-1, 1)),
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=True,
        )
    else:
        output = model(
            input_ids=top_k_ids.view(-1, 1),
            # attention_mask=torch.ones_like(top_k_ids.view(-1, 1)),
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=True,
        )
    past_key_values = output.past_key_values
    logits = output.logits[:, -1, :]
    if seq2seq:
        next_hidden = output.decoder_hidden_states[-1]
    else:
        next_hidden = output.hidden_states[-1]
    context_hidden = last_hidden_states.unsqueeze(1).expand(-1, beam_width, -1, -1).reshape(bsz * beam_width,
                                                                                            seqlen, embed_dim)

    # prepare for the classification model
    input_ids_for_class_ = torch.cat([
        input_ids_for_class.unsqueeze(1).expand(-1, beam_width, -1).reshape(bsz * beam_width, seqlen),
        top_k_ids.view(-1, 1)
    ], dim=-1
    )

    batch_text_list = []
    batch_alpha_scale_text_list = []
    for one_input_id in input_ids_for_class_:
        one_text = simctg_tokenizer.decode(one_input_id[prefix_len:][-clip_text_max_len:])
        # we only consider the class score of the generated text continuation
        batch_text_list.append(one_text)
        if alpha_scale:
            one_word = one_text.split(" ")[-vis_window_len:][0] ## consider the last vis_window_len words
            # one_word = simctg_tokenizer.decode(one_input_id[prefix_len:][-vis_window_len:]) # skip space in tokenizer
            batch_alpha_scale_text_list.append(one_word)

    batch_score = clip.compute_image_text_similarity_via_raw_text(image_embeds, batch_text_list)

    topic_batch_score = None
    if c2t and num_topic is not None:
        assert topic_classifier is not None
        topic_batch_score = topic_classifier.compute_text_probs(batch_text_list, num_topic)

    vis_window_sim = None
    if alpha_scale:
        vis_window_sim = clip.compute_image_text_similarity_via_raw_text(image_embeds,
                                                                         batch_alpha_scale_text_list)

    selected_idx = plug_and_play_fast_ranking(
        context_hidden,
        next_hidden,
        top_k_ids,
        top_k_probs,
        eta,
        alpha,
        batch_score,
        beam_width,
        condition,
        beta, ## used only with topic classifier
        topic_batch_score
    )

    # prepare for the next step
    next_id = top_k_ids[range(len(top_k_ids)), selected_idx].unsqueeze(-1)
    next_hidden = torch.stack(torch.split(next_hidden.squeeze(dim=1), beam_width))
    next_hidden = next_hidden[range(bsz), selected_idx, :]
    last_hidden_states = torch.cat([last_hidden_states, next_hidden.unsqueeze(1)], dim=1)
    past_key_values = select_past_key_values(past_key_values, beam_width, selected_idx)
    logits_selected = torch.stack(torch.split(logits, beam_width))[range(bsz), selected_idx, :]
    input_ids_for_class = torch.cat([input_ids_for_class, next_id], dim=-1)

    if seq2seq:
        returned_input_ids = input_ids
        returned_decoder_input_ids = next_id
    else:
        returned_input_ids = next_id
        returned_decoder_input_ids = None

    return returned_input_ids, returned_decoder_input_ids, past_key_values, last_hidden_states, \
        logits_selected, input_ids_for_class, logits, vis_window_sim

# ========== batch version ========= #
def ranking_fast(context_hidden, next_hidden, next_top_k_probs, eta, beam_width):
    '''
        context_hidden: bsz*beam x seqlen x embed_dim
        next_hidden: bsz*beam x 1 x embed_dim
        next_top_k_probs: bsz x beam
    '''
    _, context_len, embed_dim = context_hidden.size()
    norm_context_hidden = context_hidden / context_hidden.norm(dim=2, keepdim=True)
    norm_next_hidden = next_hidden / next_hidden.norm(dim=2, keepdim=True)
    cosine_matrix = torch.matmul(norm_context_hidden, norm_next_hidden.transpose(1,2)).squeeze(-1)    # [B*K, S]
    scores, _ = torch.max(cosine_matrix, dim=-1)    # [B*K]
    next_top_k_probs = next_top_k_probs.view(-1)    # [B*K]
    scores = (1.0 - eta) * next_top_k_probs - eta * scores
    scores = torch.stack(torch.split(scores, beam_width))    # [B, K]
    selected_idx = scores.max(dim=-1)[1]    # [B]
    return selected_idx

def ContrastiveDecodingOneStepFast(
    model,
    ids,
    beam_width,
    eta,
    past_key_values,
    last_hidden_states,
    vocab,
    logit_for_next_step,
    first_step=False,
    ):
    # input_ids: [B, S]
    if first_step:
        output = model(
            input_ids=ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True
        )
        past_key_values = output.past_key_values
        last_hidden_states = output.hidden_states[-1]    # [B, S, E]
        logit_for_next_step = output.logits[:, -1, :]    # [B, V]
    bsz, seqlen, embed_dim = last_hidden_states.size()
    p = random.uniform(0, 1)

    next_probs = F.softmax(logit_for_next_step, dim=-1)
    _, top_k_ids = torch.topk(logit_for_next_step, dim=-1, k=beam_width)    # [B, K]
    top_k_probs = torch.gather(next_probs, dim=1, index=top_k_ids)    # [B, K]
    # compute new hidden
    past_key_values = enlarge_past_key_values(past_key_values, beam_width)
    output = model(
        input_ids=top_k_ids.view(-1, 1),
        attention_mask=torch.ones_like(top_k_ids.view(-1, 1)),
        past_key_values=past_key_values,
        output_hidden_states=True,
        use_cache=True,
    )
    past_key_values = output.past_key_values
    logits = output.logits[:, -1, :]    # [B*K, V]
    next_hidden = output.hidden_states[-1]    # [B*K, 1, E]
    context_hidden = last_hidden_states.unsqueeze(1).expand(-1, beam_width, -1, -1).reshape(bsz*beam_width, seqlen, embed_dim)    # [B*K, S, E]

    selected_idx = ranking_fast(
        context_hidden,
        next_hidden,
        top_k_probs,    # [B, K]
        eta,
        beam_width,
    )     # [B]
    # prepare for the next step
    next_id = top_k_ids[range(len(top_k_ids)), selected_idx].unsqueeze(-1)    # [B, 1]
    next_hidden = torch.stack(torch.split(next_hidden.squeeze(dim=1), beam_width))    # [B, K, E]
    next_hidden = next_hidden[range(bsz), selected_idx, :]    # [B, E]
    last_hidden_states = torch.cat([last_hidden_states, next_hidden.unsqueeze(1)], dim=1)    # [B, S, E]
    past_key_values = select_past_key_values(past_key_values, beam_width, selected_idx)
    logits = torch.stack(torch.split(logits, beam_width))[range(bsz), selected_idx, :]    # [B, V]
    # next_id: [B, 1]
    return next_id, past_key_values, last_hidden_states, logits


### Loss functions
def compute_valid_token_num(valid_len_list):
    res = 0
    for one_len in valid_len_list:
        res += one_len * (one_len - 1)
    return res


def build_mask_matrix(seqlen, valid_len_list, prefix_len=0):
    '''
        prefix_len: the length of prefix that we do not want to compute CL loss for.

        (1) if a sequence of length 4 contains zero padding token (i.e., the valid length is 4),
            then the loss padding matrix looks like
                 [0., 1., 1., 1.],
                 [1., 0., 1., 1.],
                 [1., 1., 0., 1.],
                 [1., 1., 1., 0.]

        (2) if a sequence of length 4 contains 1 padding token (i.e., the valid length is 3),
            then the loss padding matrix looks like
                 [0., 1., 1., 0.],
                 [1., 0., 1., 0.],
                 [1., 1., 0., 0.],
                 [0., 0., 0., 0.]
    '''
    res_list = []
    base_mask = torch.ones(seqlen, seqlen) - torch.eye(seqlen, seqlen)
    base_mask = base_mask.type(torch.FloatTensor)
    bsz = len(valid_len_list)
    for i in range(bsz):
        one_base_mask = base_mask.clone()
        one_valid_len = valid_len_list[i]
        one_base_mask[:, one_valid_len:] = 0.
        one_base_mask[one_valid_len:, :] = 0.
        if prefix_len > 0:
            one_base_mask[:prefix_len, :prefix_len] = 0.
        res_list.append(one_base_mask)
    res_mask = torch.stack(res_list, dim=0)  # torch.FloatTensor(res_list)
    # print (res_mask)
    assert res_mask.size() == torch.Size([bsz, seqlen, seqlen])
    return res_mask


def contrastive_loss(margin, score_matrix, input_ids, pad_token_id, prefix_len=0):
    '''
       margin: predefined margin to push similarity score away
       score_matrix: bsz x seqlen x seqlen
       input_ids: bsz x seqlen
       pad_token_id: indicating which tokens are padding token
    '''
    bsz, seqlen, _ = score_matrix.size()
    gold_score = torch.diagonal(score_matrix, offset=0, dim1=1, dim2=2)  # bsz x seqlen
    gold_score = torch.unsqueeze(gold_score, -1)
    assert gold_score.size() == torch.Size([bsz, seqlen, 1])
    difference_matrix = gold_score - score_matrix
    assert difference_matrix.size() == torch.Size([bsz, seqlen, seqlen])
    loss_matrix = margin - difference_matrix  # bsz x seqlen x seqlen
    loss_matrix = torch.nn.functional.relu(loss_matrix)

    ### input mask
    input_mask = torch.ones_like(input_ids).type(torch.FloatTensor)
    if loss_matrix.is_cuda:
        input_mask = input_mask.cuda(loss_matrix.get_device())
    input_mask = input_mask.masked_fill(input_ids.eq(pad_token_id), 0.0)

    if loss_matrix.is_cuda:
        input_mask = input_mask.cuda(loss_matrix.get_device())

    valid_len_list = torch.sum(input_mask, dim=-1).tolist()
    loss_mask = build_mask_matrix(seqlen, [int(item) for item in valid_len_list], prefix_len)
    if score_matrix.is_cuda:
        loss_mask = loss_mask.cuda(score_matrix.get_device())
    masked_loss_matrix = loss_matrix * loss_mask

    loss_matrix = torch.sum(masked_loss_matrix, dim=-1)
    assert loss_matrix.size() == input_ids.size()
    loss_matrix = loss_matrix * input_mask
    cl_loss = torch.sum(loss_matrix) / torch.sum(loss_mask)
    return cl_loss
