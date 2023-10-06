# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT-2 model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.parameter import Parameter

from transformers import GPT2PreTrainedModel

# from IPython import embed

logger = logging.getLogger(__name__)


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class GPT2Config(object):
    """Configuration class to store the configuration of a `GPT2Model`.
    """

    def __init__(
            self,
            vocab_size_or_config_json_file=50257,
            n_positions=1024,
            n_ctx=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
    ):
        """Constructs GPT2Config.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `GPT2Model` or a configuration json file.
            n_positions: Number of positional embeddings.
            n_ctx: Size of the causal mask (usually same as n_positions).
            n_embd: Dimensionality of the embeddings and hidden states.
            n_layer: Number of hidden layers in the Transformer encoder.
            n_head: Number of attention heads for each attention layer in
                the Transformer encoder.
            layer_norm_epsilon: epsilon to use in the layer norm layers
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                                                               and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding="utf-8") as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.n_ctx = n_ctx
            self.n_positions = n_positions
            self.n_embd = n_embd
            self.n_layer = n_layer
            self.n_head = n_head
            self.layer_norm_epsilon = layer_norm_epsilon
            self.initializer_range = initializer_range
        else:
            raise ValueError(
                "First argument must be either a vocabulary size (int)"
                "or the path to a pretrained model config file (str)"
            )

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `GPT2Config` from a Python dictionary of parameters."""
        config = GPT2Config(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `GPT2Config` from a json file of parameters."""
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())


class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = Parameter(w)
        self.bias = Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns - nd:ns, :ns]
        w = w * b - 1e4 * (1 - b)

        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, layer_past=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            # key = key.expand(past_key.size(0), -1, -1, -1)
            key = torch.cat((past_key, key), dim=-1)
            # value = value.expand(past_value.size(0), -1, -1, -1)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1),
                               value))  # transpose to have same shapes for stacking #(stack_dim, batch, head, seq_length, head_features)
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a, present


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2


class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None):
        a, present = self.attn(self.ln_1(x), layer_past=layer_past)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present


class GPT2LMHead(nn.Module):
    """ Language Model Head for the transformer """

    def __init__(self, model_embeddings_weights, config):
        super(GPT2LMHead, self).__init__()
        self.n_embd = config.n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, hidden_state):
        # Truncated Language modeling logits (we remove the last token)
        # h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd)
        lm_logits = self.decoder(hidden_state)
        return lm_logits


class GPT2Model(GPT2PreTrainedModel):

    def __init__(self, config):
        super(GPT2Model, self).__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.init_weights()
        # self.apply(self.init_weights)

    def get_input_embeddings(self):
        return self.wte

    def forward(self, input_ids, position_ids=None, token_type_ids=None, past=None):
        # if input_ids.dtype != torch.long:
        #     input_ids = input_ids.long()

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None and input_ids.size(-1) < 20000:
            position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long,
                                        device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        elif position_ids is None and input_ids.size(-1) > 20000:
            position_ids = torch.arange(past_length, input_ids.size(-2) + past_length, dtype=torch.long,
                                        device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids[:, :, 0])
        input_shape = input_ids.size()

        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))
        flag_bb = 0

        if input_shape[-1] < 20000:
            inputs_embeds = self.wte(input_ids)
            flag_bb = 0
        else:
            input_shape = input_shape[:-1]
            inputs_embeds = torch.matmul(input_ids, self.wte.weight[:, :])
            inputs_embeds = torch.unsqueeze(inputs_embeds, dim=1)
            flag_bb = 1

        # inputs_embeds.retain_grad()
        self.i_embeds = inputs_embeds
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0

        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        presents = []
        hiddens = []
        for block, layer_past in zip(self.h, past):
            # if flag_bb == 1:
            #     print("Ran")
            #     print(hidden_states.shape)
            #     print(layer_past)
            #     exit()

            hidden_states, present = block(hidden_states, layer_past)
            hiddens.append(hidden_states)
            presents.append(present)
        hidden_states = self.ln_f(hidden_states)
        self.hiddens_list = hiddens
        self.hidden_states = hidden_states

        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states, presents

    # HACK HACK HACK
    def forward_embed(self, input_ids, position_ids=None, token_type_ids=None, past=None):
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long,
                                        device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        self.input_shape = input_ids.size()

        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds = self.wte(input_ids)
        # inputs_embeds.retain_grad()
        self.i_embeds = inputs_embeds
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        #
        # presents = []
        # for block, layer_past in zip(self.h, past):
        #     hidden_states, present = block(hidden_states, layer_past)
        #     presents.append(present)
        # hidden_states = self.ln_f(hidden_states)
        #
        # output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states

    def forward_transformer(self, hidden_states, past=None, add_one=False):
        if past is None:
            past = [None] * len(self.h)
        presents = []
        hiddens = []
        for block, layer_past in zip(self.h, past):
            hidden_states, present = block(hidden_states, layer_past)
            hiddens.append(hidden_states)
            presents.append(present)
        hidden_states = self.ln_f(hidden_states)
        self.hiddens_list = hiddens
        self.hidden_states = hidden_states
        if add_one:
            output_shape = (self.input_shape[0],) + (self.input_shape[1] + 1,) + (hidden_states.size(-1),)
        else:
            output_shape = (self.input_shape[0],) + (self.input_shape[1],) + (hidden_states.size(-1),)

        if add_one:
            present_shape = (self.input_shape[0],) + (self.input_shape[1] + 1,) + (2 * hidden_states.size(-1),)
        else:
            present_shape = (self.input_shape[0],) + (self.input_shape[1],) + (2 * hidden_states.size(-1),)

        presents = [p.view(*present_shape) for p in presents]

        return hidden_states.view(*output_shape), presents

    # def forward_hidden(self, hidden_states, past=None):
    #    if past is None:
    #        past_length = 0
    #        past = [None] * len(self.h)
    #    else:
    #        past_length = past[0][0].size(-2)
    #    presents = []
    #    for block, layer_past in zip(self.h, past):
    #        hidden_states, present = block(hidden_states, layer_past)
    #        presents.append(present)
    #    hidden_states = self.ln_f(hidden_states)

    #    #output_shape = input_shape + (hidden_states.size(-1),)
    #    #return hidden_states.view(*output_shape), presents
    #    return hidden_states, presents


class GPT2LMHeadModel(GPT2PreTrainedModel):
    authorized_missing_keys = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]
    """OpenAI GPT-2 model with a Language Modeling head ("Language Models are Unsupervised Multitask Learners").

    Params:
        config: a GPT2Config class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length] (or more generally [d_1, ..., d_n, sequence_length]
            were d_1 ... d_n are arbitrary dimensions) with the word BPE token indices selected in the range [0, config.vocab_size[
        `position_ids`: an optional torch.LongTensor with the same shape as input_ids
            with the position indices (selected in the range [0, config.n_positions - 1[.
        `token_type_ids`: an optional torch.LongTensor with the same shape as input_ids
            You can use it to add a third type of embedding to each input token in the sequence
            (the previous two being the word and position embeddings).
            The input, position and token_type embeddings are summed inside the Transformer before the first
            self-attention block.
        `lm_labels`: optional language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `past`: an optional list of torch.LongTensor that contains pre-computed hidden-states
            (key and values in the attention blocks) to speed up sequential decoding
            (this is the presents output of the model, cf. below).

    Outputs:
        if `lm_labels` is not `None`:
            Outputs the language modeling loss.
        else a tuple:
            `lm_logits`: the language modeling logits as a torch.FloatTensor of size [batch_size, sequence_length, config.vocab_size]
                (or more generally [d_1, ..., d_n, config.vocab_size] were d_1 ... d_n are the dimension of input_ids)
            `presents`: a list of pre-computed hidden-states (key and values in each attention blocks) as
                torch.FloatTensors. They can be reused to speed up sequential decoding.

    Example usage:
    ```python
    # Already been converted into BPE token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])

    config = modeling_gpt2.GPT2Config()

    model = modeling_gpt2.GPT2LMHeadModel(config)
    lm_logits, presents = model(input_ids)
    ```
    """

    def __init__(self, config):
        super(GPT2LMHeadModel, self).__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, position_ids=None, token_type_ids=None, lm_labels=None, past=None):
        hidden_states, presents = self.transformer(input_ids, position_ids, token_type_ids, past)

        self.hidden_states = hidden_states
        lm_logits = self.lm_head(hidden_states)
        if lm_labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[:, :-1].contiguous()
            shift_labels = lm_labels[:, 1:].contiguous()

            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            return loss
        return lm_logits, presents

    # HACK HACK HACK
    def forward_embed(self, inputs_ids, position_ids=None, token_type_ids=None, past=None):
        hidden_states = self.transformer.forward_embed(inputs_ids, position_ids, token_type_ids, past)
        #     self.hidden_states_fe = hidden_states
        #     lm_logits = self.lm_head(hidden_states)
        return hidden_states

    def forward_transformer_embed(self, hidden_states, past=None, add_one=False):
        hidden_states, presents = self.transformer.forward_transformer(hidden_states, past, add_one=add_one)
        # lm_logits = self.lm_head(hidden_states)
        # if lm_labels is not None:
        #     # Shift so that tokens < n predict n
        #     shift_logits = lm_logits[:, :-1].contiguous()
        #     shift_labels = lm_labels[:, 1:].contiguous()
        #
        #     # Flatten the tokens
        #     loss_fct = CrossEntropyLoss(ignore_index=-1)
        #     loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
        #                     shift_labels.view(-1))
        #     return loss
        return hidden_states, presents

    def forward_hidden(self, hidden_states):
        # hidden_states, presents = self.transformer.forward_hidden(hidden_states, past)
        # self.hidden_states_fh = hidden_states
        '''Just runing the last MLP (LM head)'''
        lm_logits = self.lm_head(hidden_states)
        return lm_logits

