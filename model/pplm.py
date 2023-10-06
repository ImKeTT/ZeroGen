#! /usr/bin/env python3
# coding=utf-8

# This code is licensed under a non-commercial license.

import os
import sys
import argparse
from tqdm import trange, tqdm
import json, datetime, random

import torch
import torch.nn.functional as F
import numpy as np
from operator import add
sys.path.append("../../")
from ctgsrc.model.clip import CLIP
from PIL import Image

# sys.path.append("..")
# from classifier.annotator import Attn, MLP
#
# from gpt2tunediscrim import ClassificationHead

# lab_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..')
# sys.path.insert(1, lab_root)

from transformers import GPT2Tokenizer, AutoTokenizer
from modeling_gpt2 import GPT2LMHeadModel

SmallConst = 1e-15

GEN_PREFIX_PATH="/mnt/data0/tuhq21/news_writer/ctgsrc/pplm_generated"

model_name = "/mnt/data0/tuhq21/news_writer/ctgsrc/language_model/output/visnews_long_dev_ppl_14.591"
sos_token, pad_token = r'<-start_of_text->', r'<-pad->'
eos_token = r'<|endoftext|>'

def add_special_token(special_token, tokenizer):
    if special_token in tokenizer.vocab:
        print(special_token + ' token exists.')
    else:
        print('Add token to the tokenizer.')
        print('Original vocabulary size is {}'.format(len(tokenizer)))
        tokenizer.add_tokens([special_token])
        print('Vocabulary size after extension is {}'.format(len(tokenizer)))
        assert len(tokenizer.convert_tokens_to_ids([special_token])) == 1
    special_token_id = tokenizer.convert_tokens_to_ids([special_token])[0]
    return tokenizer, special_token, special_token_id

model = GPT2LMHeadModel.from_pretrained(model_name)
enc = AutoTokenizer.from_pretrained(model_name)
enc, sos_token, sos_token_id = add_special_token(sos_token, enc)
enc, pad_token, pad_token_id = add_special_token(pad_token, enc)

tot_tendency = []

# This code is licensed under a non-commercial license.

from torch.autograd import Variable

def to_var(x, requires_grad=False, volatile=False, device='cuda'):
    x = x.to(device)
    return Variable(x, requires_grad=requires_grad, volatile=volatile)

def top_k_logits(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins, torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)


def perturb_past(past, model, prev, args, classifier, good_index=None, stepsize=0.01, vocab_size=len(enc),
                 original_probs=None, accumulated_hidden=None, true_past=None, grad_norms=None, alter_scale=1.0,
                 device='cuda'):
    window_length = args.window_length
    gm_scale, kl_scale = args.fusion_gm_scale, args.fusion_kl_scale
    one_hot_vectors = []
    for good_list in good_index:
        good_list = list(filter(lambda x: len(x) <= 1, good_list))
        # print(good_list)
        good_list = torch.tensor(good_list).to(device)
        num_good = good_list.shape[0]
        one_hot_good = torch.zeros(num_good, vocab_size).to(device)
        one_hot_good.scatter_(1, good_list, 1)
        one_hot_vectors.append(one_hot_good)

    # Generate inital perturbed past
    past_perturb_orig = [(np.random.uniform(0.0, 0.0, p.shape).astype('float32'))
                         for p in past]

    if accumulated_hidden is None:
        accumulated_hidden = 0

    if args.decay:
        decay_mask = torch.arange(0., 1.0 + SmallConst, 1.0 / (window_length))[1:]
    else:
        decay_mask = 1.0

    # Generate a mask is gradient perturbated is based on a past window
    _, _, _, current_length, _ = past[0].shape

    if current_length > window_length and window_length > 0:
        ones_key_val_shape = tuple(past[0].shape[:-2]) + tuple([window_length]) + tuple(
            past[0].shape[
            -1:])  # (stack_dim, batch, head, seq_length, head_features) -> (stack_dim, batch, head, window_length, head_features)

        zeros_key_val_shape = tuple(past[0].shape[:-2]) + tuple([current_length - window_length]) + tuple(
            past[0].shape[
            -1:])  # (stack_dim, batch, head, seq_length, head_features) -> (stack_dim, batch, head, seq_length - window_length, head_features)

        ones_mask = torch.ones(ones_key_val_shape)
        ones_mask = decay_mask * ones_mask.permute(0, 1, 2, 4, 3)
        ones_mask = ones_mask.permute(0, 1, 2, 4, 3)

        window_mask = torch.cat((ones_mask, torch.zeros(zeros_key_val_shape)), dim=-2).to(device)
    else:
        window_mask = torch.ones_like(past[0]).to(device)

    loss_per_iter = []
    for i in range(args.num_iterations):
        past_perturb = [torch.from_numpy(p_) for p_ in past_perturb_orig]
        past_perturb = [to_var(p_, requires_grad=True, device=device) for p_ in past_perturb]

        perturbed_past = list(map(add, past, past_perturb))

        _, _, _, current_length, _ = past_perturb[0].shape

        # Compute hidden using perturbed past
        _, future_past = model(prev, past=perturbed_past)
        hidden = model.hidden_states
        new_accumulated_hidden = accumulated_hidden + torch.sum(hidden, dim=1).detach()

        # TODO: Check the layer-norm consistency of this with trained discriminator
        logits = model.forward_hidden(hidden)
        logits = logits[:, -1, :]
        probabs = F.softmax(logits, dim=-1)
        loss = 0.0
        loss_list = []
        if args.loss_type == 1 or args.loss_type == 3:
            for one_hot_good in one_hot_vectors:
                good_logits = torch.mm(probabs, torch.t(one_hot_good))
                loss_word = good_logits
                loss_word = torch.sum(loss_word)
                loss_word = -torch.log(loss_word)
                # loss_word = torch.sum(loss_word) /torch.sum(one_hot_good)
                loss += loss_word
                loss_list.append(loss_word)
            if args.print_intermediate_result:
                print('words', loss.data.cpu().numpy())

        if args.loss_type == 2 or args.loss_type == 3:
            ce_loss = torch.nn.CrossEntropyLoss()
            new_true_past = true_past
            for i in range(args.horizon_length):
                future_probabs = F.softmax(logits, dim=-1)  # Get softmax
                future_probabs = torch.unsqueeze(future_probabs, dim=1)
                # future_probabs.shape[-1] == vocab_size

                _, new_true_past = model(future_probabs, past=new_true_past)
                future_hidden = model.hidden_states  # Get expected hidden states
                new_accumulated_hidden = new_accumulated_hidden + torch.sum(future_hidden, dim=1)

            predicted_sentiment = classifier(new_accumulated_hidden / (current_length + 1 + args.horizon_length))

            label = torch.tensor([args.label_class], device=device, dtype=torch.long)
            discrim_loss = ce_loss(predicted_sentiment, label)
            if args.print_intermediate_result:
                print('discrim', discrim_loss.data.cpu().numpy())
            loss += discrim_loss
            loss_list.append(discrim_loss)

        kl_loss = 0.0
        if kl_scale > 0.0:
            p = (F.softmax(original_probs[:, -1, :], dim=-1))
            p = p + SmallConst * (p <= SmallConst).type(torch.FloatTensor).to(device).detach()
            correction = SmallConst * (probabs <= SmallConst).type(torch.FloatTensor).to(device).detach()
            corrected_probabs = probabs + correction.detach()
            kl_loss = kl_scale * ((corrected_probabs * (corrected_probabs / p).log()).sum())
            # print('kl_loss', kl_loss.data.cpu().numpy())
            loss += kl_loss  # + discrim_loss
        if args.print_intermediate_result:
            print((loss - kl_loss).data.cpu().numpy())

        loss_per_iter.append(loss.data.cpu().numpy())
        loss.backward()
        if grad_norms is not None and args.loss_type == 1:
            grad_norms = [torch.max(grad_norms[index], torch.norm(p_.grad * window_mask)) for index, p_ in
                          enumerate(past_perturb)]
        else:
            grad_norms = [(torch.norm(p_.grad * window_mask) + SmallConst) for index, p_ in enumerate(past_perturb)]

        grad = [
            -stepsize * alter_scale * (p_.grad * window_mask / grad_norms[index] ** args.gamma).data.cpu().numpy()
            for index, p_ in enumerate(past_perturb)]
        past_perturb_orig = list(map(add, grad, past_perturb_orig))

        for p_ in past_perturb:
            p_.grad.data.zero_()

        new_past = []
        for p in past:
            new_past.append(p.detach())

        past = new_past

    past_perturb = [torch.from_numpy(p_) for p_ in past_perturb_orig]
    past_perturb = [to_var(p_, requires_grad=True, device=device) for p_ in past_perturb]
    perturbed_past = list(map(add, past, past_perturb))

    return perturbed_past, new_accumulated_hidden, grad_norms, loss_per_iter

def enlarge_past_key_values(past_key_values, beam_width):
    # from [B, num_head, seq_len, esz] to [B*K, num_head, seq_len, esz]
    new_key_values = []
    for layer in past_key_values:
        items = ()
        for item in layer:
            # item is the key and value matrix
            bsz, num_head, seq_len, esz = item.size()
            item = item.unsqueeze(1).expand(-1, beam_width, -1, -1, -1).reshape(bsz*beam_width, num_head, seq_len, esz)    # [bsz*beam, num_head, seq_len, esz]
            items+=(item.unsqueeze(0),)
        items = torch.cat(items, dim=0)
        new_key_values.append(items)
    return new_key_values

def latent_perturb(model, args, context=None, sample=True, device='cuda', image_embeds=None, prefix_len=0):
    bow_index = None
    classifier = None
    annotator = None

    # Get tokens for the list of positive words
    def list_tokens(word_list):
        token_list = []
        for word in word_list:
            token_list.append(enc.encode(" " + word))
        return token_list

    good_index = []
    if args.bag_of_words:
        bags_of_words = args.bag_of_words.split(";")
        for wordlist in bags_of_words:
            with open("../wordlists/" + wordlist + ".txt", "r") as f:
                words = f.read()
                words = words.split('\n')
            good_index.append(list_tokens(words))

    if args.bag_of_words and classifier:
        if args.print_intermediate_result:
            print('Both PPLM-BoW and PPLM-Discrim are on. This is not optimized.')
        args.loss_type = 3

    elif args.bag_of_words:
        args.loss_type = 1
        if args.print_intermediate_result:
            print('Using PPLM-BoW')

    elif classifier is not None:
        args.loss_type = 2
        if args.print_intermediate_result:
            print('Using PPLM-Discrim')

    else:
        raise Exception('Supply either --bag-of-words (-B) or --discrim -D')

    if bow_index is not None:
        good_index = [bow_index]

    if args.require_origin:
        original, _, _ = sample_from_hidden(model=model, args=args, context=context, device=device,
                                            sample=sample, perturb=False, good_index=good_index, classifier=classifier,
                                            annotator=annotator, image_embeds=image_embeds, prefix_len=prefix_len)
    if device != 'cpu':
        torch.cuda.empty_cache()

    perturbed_list = []
    discrim_loss_list = []
    loss_in_time_list = []

    for i in range(args.num_samples):
        perturbed, discrim_loss, loss_in_time = sample_from_hidden(model=model, args=args, context=context,
                                                                   device=device, sample=sample, perturb=True,
                                                                   good_index=good_index,
                                                                   classifier=classifier, annotator=annotator,
                                                                   image_embeds=image_embeds, prefix_len=prefix_len)
        perturbed_list.append(perturbed)
        if classifier is not None:
            discrim_loss_list.append(discrim_loss.data.cpu().numpy())
        loss_in_time_list.append(loss_in_time)

    if device != 'cpu':
        torch.cuda.empty_cache()

    if args.require_origin:
        return original, perturbed_list, discrim_loss_list, loss_in_time_list
    else:
        return perturbed_list, discrim_loss_list, loss_in_time_list

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

def sample_from_hidden(model, args, classifier, context=None, past=None, device='cuda',
                       sample=True, perturb=True, good_index=None, annotator=None,
                       image_embeds=None, prefix_len=0):
    output = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0) if context else None

    def exam_BOW_distribution(good_index, log_probs):
        # good_index = [[input_id1], [inputid2], ...]
        ans = []
        for indices in good_index:

            sum = 0
            for ids in indices:
                sum += log_probs[0][ids[0]]

            ans.append(sum.item())
        return ans

    def exam_Disc_distribution(true_hidden, annotator, temperature=0.5):
        probs = F.softmax(annotator(true_hidden) / temperature, dim=-1)[:, -1, :].to('cpu')
        size = probs.shape[-1]
        dist = torch.tensor(range(size)) * (1 / size) + (0.5 / size)
        if args.discrim == 'sentiment':
            res = torch.abs(torch.sum(probs * dist, dim=-1).squeeze() - 0.5).item()

        elif args.discrim == 'toxicity':
            res = torch.sum(probs * dist, dim=-1).squeeze().item()

        elif args.discrim == 'clickbait':
            res = torch.sum(probs * dist, dim=-1).squeeze().item()
        # print(res)
        # raise Exception

        return res

    perplexity = 0.0
    length = 0
    tendency_sit = [0] * len(good_index)

    if args.add_vis and image_embeds is not None:
        clip = CLIP(args.clip_name, device)
        image_embeds = clip.compute_image_representation_from_image_instance(image_embeds)
    else:
        clip = None

    grad_norms = None
    loss_in_time = []
    for i in trange(args.length, ascii=True):

        # output = output.expand(args.beam_width, -1)
        # Get past/probs for current output, except for last word
        # Note that GPT takes 2 inputs: past + current-token
        # Therefore, use everything from before current i/p token to generate relevant past
        # output = output.expand(args.beam_width, -1)
        if past is None and output is not None:
            prev = output
            _, past = model(output[:, :-1])  # _, past = loss_of_GPT2LMHead, [torch.stack([key, value])] * block_layer
            original_probs, true_past = model(output)
            true_hidden = model.hidden_states
            logits = model.forward_hidden(true_hidden)
            logits = logits[:, -1, :] / args.temperature  # + SmallConst
            bsz, seqlen, embed_dim = true_hidden.size()
        else:
            original_probs, true_past = model(output)
            true_past = enlarge_past_key_values(true_past, args.beam_width)
            true_hidden = model.hidden_states

        # Modify the past if necessary

        if i >= args.grad_length:
            current_stepsize = args.stepsize * 0
        else:
            current_stepsize = args.stepsize

        if perturb:
            tmp_original_probs = F.softmax(original_probs[:, -1, :], dim=-1)
            if args.activate_alter_scale and args.bag_of_words:
                alter_scale = np.array(exam_BOW_distribution(good_index, tmp_original_probs)).mean() / args.activesize

            elif args.activate_alter_scale and classifier:
                if args.annotator_type == 'dis':
                    alter_scale = exam_Disc_distribution(true_hidden, annotator) / args.activesize
                    # alter_scale = 1.0
                elif args.annotator_type == 'bow':
                    alter_scale = np.array(exam_BOW_distribution(good_index, tmp_original_probs)).mean() / (
                                2 * args.activesize)

            else:
                alter_scale = 1.0

        if clip is not None:
            ## args: beam_width; prefix_len; clip_text_max_len;
            _, top_k_ids = torch.topk(logits, dim=-1, k=args.beam_width)
            input_ids_for_class_ = torch.cat([
                output.unsqueeze(1).expand(-1, args.beam_width, -1).reshape(1 * args.beam_width, -1),
                top_k_ids.view(-1, 1)
            ], dim=-1
            )
            batch_text_list = []
            batch_beta_scale_text_list = []
            for one_input_id in input_ids_for_class_:
                one_text = enc.decode(one_input_id[prefix_len:][-args.clip_text_max_len:])
                # we only consider the class score of the generated text continuation
                batch_text_list.append(one_text)

                if args.beta_scale:
                    one_word = one_text.split(" ")[-args.vis_window_len:][0]  ## consider the last vis_window_len words
                    # one_word = simctg_tokenizer.decode(one_input_id[prefix_len:][-vis_window_len:]) # skip space in tokenizer
                    batch_beta_scale_text_list.append(one_word)
                    vis_window_sim = clip.compute_image_text_similarity_via_raw_text(image_embeds,
                                                                                     batch_beta_scale_text_list)

            batch_score = clip.compute_image_text_similarity_via_raw_text(image_embeds, batch_text_list)

        if not perturb or args.num_iterations == 0:
            past = enlarge_past_key_values(past, args.beam_width)
            perturbed_past = past
        else:
            accumulated_hidden = model.hidden_states[:, :-1, :]  # [bsz, seq_length, dimension]
            accumulated_hidden = torch.sum(accumulated_hidden, dim=1)
            get_in = top_k_ids.view(-1, 1) #prev if i == 0 else top_k_ids.view(-1, 1)
            # if i:
            past = enlarge_past_key_values(past, args.beam_width)
            perturbed_past, _, grad_norms, loss_per_iter = perturb_past(past, model, get_in, args,
                                                                        good_index=good_index,
                                                                        stepsize=current_stepsize,
                                                                        original_probs=original_probs,
                                                                        true_past=true_past,
                                                                        accumulated_hidden=accumulated_hidden,
                                                                        classifier=classifier,
                                                                        grad_norms=grad_norms,
                                                                        alter_scale=alter_scale,
                                                                        device=device)

            loss_in_time.append(loss_per_iter)
        get_in = top_k_ids.view(-1, 1) #prev if i == 0 else top_k_ids.view(-1, 1)
        test_logits, past = model(get_in, past=perturbed_past)

        # test_logits = F.softmax(test_logits[:, -1, :], dim=-1)
        # likelywords = torch.topk(test_logits, k=10, dim=-1)
        # print(enc.decode(likelywords[1].tolist()[0]))

        if classifier is not None:
            ce_loss = torch.nn.CrossEntropyLoss()
            predicted_sentiment = classifier(torch.mean(true_hidden, dim=1))
            label = torch.tensor([args.label_class], device='cuda', dtype=torch.long)
            true_discrim_loss = ce_loss(predicted_sentiment, label)
            if args.print_intermediate_result:
                print("true discrim loss", true_discrim_loss.data.cpu().numpy())
        else:
            true_discrim_loss = 0

        hidden = model.hidden_states  # update hidden
        bsz, seqlen, embed_dim = hidden.size()
        logits = model.forward_hidden(hidden)
        logits = logits[:, -1, :] / args.temperature  # + SmallConst


        # logits = top_k_logits(logits, k=args.top_k)  # + SmallConst

        log_probs = F.softmax(logits, dim=-1)

        # Fuse the modified model and original model
        if perturb:
            # original_probs = top_k_logits(original_probs[:, -1, :]) #+ SmallConst
            # tmp_original_probs = F.softmax(original_probs[:, -1, :], dim=-1)
            # likelywords = torch.topk(original_probs, k=10, dim=-1)
            # print(enc.decode(likelywords[1].tolist()[0]))
            if args.print_intermediate_result and args.bag_of_words:
                ori_tokens = [enc.decode([tmp]) for tmp in torch.topk(tmp_original_probs, k=args.top_k)[1].tolist()[0]]
                print("Original Distribution: " + str(ori_tokens))
                if good_index is not None:
                    print("Original Style Tendency: " + str(exam_BOW_distribution(good_index, tmp_original_probs)))
                per_tokens = [enc.decode([tmp]) for tmp in torch.topk(log_probs, k=args.top_k)[1].tolist()[0]]
                print("Perturbed Distribution: " + str(per_tokens))
                if good_index is not None:
                    print("Perturbed Style Tendency: " + str(exam_BOW_distribution(good_index, log_probs)))

            gm_scale = args.fusion_gm_scale
            if clip is not None:
                if args.beta_scale:
                    alter_scale = vis_window_sim.mean(-1) / args.beta_activesize
                    beta = args.beta * alter_scale
                    beta = max(beta, args.beta_upper)
                else:
                    beta = args.beta
                log_probs += beta * batch_score.view([args.beam_width, 1])
                log_probs = log_probs / torch.sum(log_probs)
            log_probs = ((log_probs * gm_scale) + (tmp_original_probs * (1 - gm_scale)))  # + SmallConst

            if args.print_intermediate_result and args.bag_of_words:
                gm_tokens = [enc.decode([tmp]) for tmp in torch.topk(log_probs, k=args.top_k)[1].tolist()[0]]
                print("GM Combined Distribution: " + str(gm_tokens))
                if good_index is not None:
                    print("GM Combined Style Tendency: " + str(exam_BOW_distribution(good_index, log_probs)))

            log_probs = top_k_logits(log_probs, k=args.top_k, probs=True)  # + SmallConst

            log_probs = log_probs / torch.sum(log_probs)

        else:
            logits = top_k_logits(logits, k=args.top_k)  # + SmallConst
            log_probs = F.softmax(logits, dim=-1)
        if sample:
            # likelywords = torch.topk(log_probs, k=args.top_k, dim=-1)
            # print(enc.decode(likelywords[1].tolist()[0]))
            # print(likelywords[0].tolist())
            prev = torch.multinomial(log_probs, num_samples=1)
            prev_selected = 0
        else:
            prev_selected = log_probs.max(dim=-1)[0].max(dim=-1)[1]
            _, prev = torch.topk(log_probs, k=1, dim=-1)
        # if perturb:
        #     prev = future

        # if i:
        prev = top_k_ids[range(len(top_k_ids)), prev_selected].unsqueeze(-1)
        # prev = prev_selected.unsqueeze(-1)
        output = prev if output is None else torch.cat((output, prev), dim=1)  # update output
        logits = torch.stack(torch.split(logits, args.beam_width))[range(1), prev_selected, :]
        past = select_past_key_values(past, args.beam_width, prev_selected)
        # output = prev if output is None else torch.cat((output, prev), dim=1)  # update output
        if args.print_intermediate_result:
            print(enc.decode(output.tolist()[0]))

        # print("PerPLexity: " + str(torch.exp(-perplexity/length).item()))
        # output_file.write("Strength: " + str(tendency_sit) + '\n')
        # print(perplexity/length)
        # raise Exception
    return output, true_discrim_loss, loss_in_time

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

def run_model(args):
    now = datetime.datetime.now()
    setup_seed(args.seed)
    device = 'cpu' if args.nocuda else 'cuda'
    model.to(device)

    # Freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False
    pass

    def get_prompt_id(text, tokenizer):
        text = text
        tokens = tokenizer.tokenize(text)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        return input_ids

    vis_append = f"_vis" if args.add_vis else ""
    beta_scale_append = f"_beta_scale{args.beta_upper}" if args.beta_scale else ""
    save_name = f"pplm-{args.bag_of_words}{vis_append}-beta{args.beta}{beta_scale_append}-gamma{args.gamma}" \
                f"-bw{args.beam_width}-iter{args.num_iterations}-gm{args.fusion_gm_scale}-" \
                f"wl{args.window_length}-{now.month}.{now.day}.json"
    save_path_prefix = GEN_PREFIX_PATH

    if os.path.exists(save_path_prefix):
        pass
    else:  # recursively construct directory
        os.makedirs(save_path_prefix, exist_ok=True)
    full_save_path = os.path.join(GEN_PREFIX_PATH, save_name)
    print('full save path is {}'.format(full_save_path))

    with open(args.test_path) as f:
        item_list = json.load(f)

    test_num = 200 #len(item_list)
    # test_num = 3
    print('Number of inference instances is {}'.format(test_num))
    time_list = []
    result_list = []
    for p_idx in tqdm(range(test_num)):
        one_test_dict = item_list[p_idx]
        one_res_dict = {
            'split': one_test_dict['split'],
            'topic': one_test_dict['topic'] if 'topic' in one_test_dict.keys() else 'none',
            'image_name': one_test_dict['image_name'],
            'captions': one_test_dict['captions'],
            'title': one_test_dict['title'] if 'title' in one_test_dict.keys() else 'none',
            'obj': 'none'
        }
        one_res_dict['prediction'] = []
        one_res_dict['time'] = []
        if args.uncond:
            seq = eos_token
        else:
            seq = f"{sos_token} {one_res_dict['title']} {eos_token}"

        input_ids = get_prompt_id(seq, enc)
        prefix_len = len(input_ids)

        # input_ids = [50256, 50256]
        # prefix_len = 2

        image_full_path = args.test_image_prefix_path + '/' + one_test_dict['image_name']

        image_instance = Image.open(image_full_path)

        bag_of_words = [args.bag_of_words]
        collect_gen = dict()
        current_index = 0
        for tmp_bow in bag_of_words:
            start_time = datetime.datetime.now()
            args.bag_of_words = tmp_bow
            # print(args.bag_of_words)
            res = []

            # text = enc.decode(out)
            # if args.print_result:
            #    print("=" * 40 + " Prefix of sentence " + "=" * 40)
            #    print(text)
            #    print("=" * 80)
            if args.require_origin:
                out1, out_perturb, discrim_loss_list, loss_in_time_list = latent_perturb(model=model, args=args,
                                                                                         context=input_ids,
                                                                                         sample=args.sample,
                                                                                         device=device,
                                                                                         image_embeds=image_instance,
                                                                                         prefix_len=prefix_len
                                                                                         )
            else:
                out_perturb, discrim_loss_list, loss_in_time_list = latent_perturb(model=model, args=args,
                                                                                   context=input_ids,
                                                                                   sample=args.sample,
                                                                                   device=device,
                                                                                   image_embeds=image_instance,
                                                                                   prefix_len=prefix_len
                                                                                   )

            if args.require_origin:
                text_whole = enc.decode(out1.tolist()[0])
            # if args.print_result:
            #    print("=" * 80)
            #    print("=" * 40 + " Whole sentence (Original)" + "=" * 40)
            #    print(text_whole)
            #    print("=" * 80)

            out_perturb_copy = out_perturb
            generated = 0
            end_time = datetime.datetime.now()
            time_diff = (end_time - start_time)
            execution_time = time_diff.total_seconds() * 1000

            for out_perturb in out_perturb_copy:
                # try:
                # if args.print_result:
                #    print("=" * 40 + " Whole sentence (Perturbed)" + "=" * 40)
                text_whole = enc.decode(out_perturb.tolist()[0][prefix_len:], skip_special_tokens=True).strip()
                one_res_dict['prediction'].append(text_whole)
                one_res_dict['time'].append(execution_time)
                res.append(text_whole)
                    # if args.print_result:
                    #    print(text_whole)
                    #    print("=" * 80)

                # collect_gen[current_index] = [out, out_perturb, out1]
                # Save the prefix, perturbed seq, original seq for each index

                current_index = current_index + 1

            if tmp_bow is not None:
                collect_gen[str(tmp_bow) + str(int(10000 * args.stepsize * args.num_iterations) / 100)] = res
            else:
                tmp_label = 'None'
                if args.discrim == 'clickbait':
                    if args.label_class == 1:
                        tmp_label = 'clickbaity'
                elif args.discrim == 'sentiment':
                    if args.label_class == 3:
                        tmp_label = 'Negative'
                    elif args.label_class == 2:
                        tmp_label = 'Positive'
                elif args.discrim == 'toxicity':
                    if args.label_class == 0:
                        tmp_label = 'nontoxic'
                collect_gen[tmp_label + str(int(10000 * args.stepsize * args.num_iterations) / 100)] = res
        result_list.append(one_res_dict)
    with open(full_save_path, 'w', encoding='utf8') as outfile:
        json.dump(result_list, outfile, indent=4, ensure_ascii=False)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', '-M', type=str, default='gpt2-medium',
                    help='pretrained model name or path to local checkpoint')
parser.add_argument('--bag-of-words', '-B', type=str, default=None,
                    help='Bags of words used for PPLM-BoW. Multiple BoWs separated by ;')
parser.add_argument('--discrim', '-D', type=str, default=None,
                    choices=('clickbait', 'sentiment', 'toxicity'),
                    help='Discriminator to use for loss-type 2')
parser.add_argument('--label-class', type=int, default=-1, help='Class label used for the discriminator')
parser.add_argument('--stepsize', type=float, default=0.02)
parser.add_argument("--length", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_k", type=int, default=10)
parser.add_argument("--fusion-gm-scale", type=float, default=0.9)
parser.add_argument("--fusion-kl-scale", type=float, default=0.01)
parser.add_argument('--nocuda', action='store_true', help='no cuda')
parser.add_argument('--uncond', action='store_true', help='Generate from end-of-text as prefix')
parser.add_argument("--cond-text", type=str, default='The lake', help='Prefix texts to condition on')
parser.add_argument('--num-iterations', type=int, default=3)
parser.add_argument('--grad-length', type=int, default=10000)
parser.add_argument('--num-samples', type=int, default=1,
                    help='Number of samples to generate from the modified latents')
parser.add_argument('--horizon-length', type=int, default=1, help='Length of future to optimize over')
# parser.add_argument('--force-token', action='store_true', help='no cuda')
parser.add_argument('--window-length', type=int, default=0,
                    help='Length of past which is being optimizer; 0 corresponds to infinite window length')
parser.add_argument('--decay', action='store_true', help='whether to decay or not')
parser.add_argument('--gamma', type=float, default=1.5)
parser.add_argument(
    "--sample", action="store_true",
    help="Generate from end-of-text as prefix"
)
parser.add_argument('--activate-alter-scale', action="store_true")
parser.add_argument('--print-result', action="store_true")
parser.add_argument('--print-intermediate-result', action="store_true")
parser.add_argument('--require-origin', action="store_true", help="Calculate origin distribution")
parser.add_argument('--activesize', type=float, default=0.01)
parser.add_argument('--classifier-type', type=str, default='attn', choices=('attn', 'mlp'))
parser.add_argument('--annotator-type', type=str, default='bow', choices=('bow', 'dis'))

## add visual control args: beam_width; prefix_len; clip_text_max_len;
parser.add_argument('--add_vis', action='store_true')
parser.add_argument('--beta_scale', action='store_true')
parser.add_argument('--beta_upper', type=float, default=1.0)
parser.add_argument('--beam_width', type=int, default=5)
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--clip_text_max_len', type=int, default=50)
parser.add_argument('--test_path', type=str)
parser.add_argument('--clip_name', type=str, default="openai/clip-vit-base-patch32")
parser.add_argument('--vis_window_len', type=int, default=1)
parser.add_argument('--beta_activesize', type=float, default=0.2)



# CUDA_VISIBLE_DEVICES=0 python pplm.py -B military --cond-text "The potato" --length 50 --gamma 1.5 --num-iterations 3 --num-samples 10 --stepsize 0.03 --window-length 5 --fusion-kl-scale 0.01 --fusion-gm-scale 0.99 --sample
if __name__ == '__main__':
    args = parser.parse_args()
    # args = parser.parse_args('-B negative_words_s --length 130 --gamma 1.5 --num-iterations 3 --beam_width 5 --beta 1.0 '
    #                          '--num-samples 1 --stepsize 0.03 --window-length 5 --fusion-kl-scale 0.05 --fusion-gm-scale 0.90 --add_vis --beta_scale --beta_upper 8.0'.split())
    dataprefix = "/mnt/data0/tuhq21/dataset/visnews/origin"
    args.test_path = os.path.join(dataprefix, "simctg_test_news.json")
    args.clip_vis_pkl = None
    args.test_image_prefix_path = dataprefix

    run_model(args)
