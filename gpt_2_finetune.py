#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
@file: gpt_2_finetune.py
@author: ImKe at 2022/8/3
@email: tuisaac163@gmail.com
@feature: #Enter features here
"""
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import torch
from tensorboardX import SummaryWriter
from logger import Logger
from torch.optim import Adam
from tqdm import tqdm
import torch.nn as nn
import datetime, math, os, sys, json, argparse, time, re
import numpy as np
from torch.utils.data import Dataset, DataLoader

CACHE_DIR='/data/tuhq/.cache/torch/transformers'
DATA_PREFIX="/data/tuhq/multimodal/coco/annotations"

os.environ["CUDA_VISIBLE_DEVICES"]="1"
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='coco', type=str, required=False)

parser.add_argument("--seed", default=42, type=int, required=False)

parser.add_argument("--load", default=None, type=str, required=False)
parser.add_argument("--num_train_epochs", default=30, type=int, required=False)
parser.add_argument("--learning_rate", default=1e-4, type=float, required=False)
parser.add_argument("--weight_decay", default=1e-6, type=float, required=False)
parser.add_argument("--early_stop", default=3, type=int, required=False)
parser.add_argument("--log_epoch", default=2, type=int, required=False)
parser.add_argument("--test_iter", default=300, type=int, required=False)
parser.add_argument("--output_dir", default="output", type=str, required=False)
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

parser.add_argument('--no_gpu', action='store_true')
parser.add_argument('--gpu', nargs='+', type=int, default=[0])
parser.add_argument("--per_gpu_train_batch_size", default=50, type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                    help="Batch size per GPU/CPU for evaluation.")
parser.add_argument("--workers", default=3, type=int,
                    help="Dataloader worker.")

class COCOCaption(Dataset):
    def __init__(self, caps):
        self.caps = caps
        self._initialization()

    def _initialization(self):
        self.textual_caps = [line['caption'].strip() for line in self.caps['annotations']]

    def __getitem__(self, item):
        cap = self.textual_caps[item]
        return cap

    def __len__(self):
        return len(self.textual_caps)


def evaluate(args, val_loader, model, tokenizer, logging, device):
    model.eval()
    losses = []
    for i, data_dict in enumerate(tqdm(val_loader)):
        encoded = tokenizer(data_dict, padding=True, return_tensors="pt")
        inputs = {
            "input_ids": encoded['input_ids'].to(device),
            "attention_mask": encoded['attention_mask'].to(device),
            "labels": encoded["input_ids"].to(device)
        }
        outputs = model(**inputs)
        loss = outputs[0]
        losses.append(loss.item())

    logging.info("========================= Testing Status =========================")
    logging.info("Avg. Val Loss            : {:.4f}".format(loss.item()))
    logging.info("===================================================================")
    return loss.item()

def run(args):
    now = datetime.datetime.now()
    # GPU
    # if not torch.cuda.is_available(): args.no_gpu = True
    gpu = not args.no_gpu
    if gpu:
        print("There are ", torch.cuda.device_count(), " available GPUs!")
        print('Current single GPU: {}'.format(torch.cuda.current_device()))
        args.n_gpu = len(args.gpu)
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    else:
        args.train_batch_size = args.per_gpu_train_batch_size
        args.eval_batch_size = args.per_gpu_eval_batch_size
    device = torch.device(args.gpu[0] if gpu else "cpu")

    # randomness
    np.random.seed(args.seed)
    prng = np.random.RandomState()
    torch.random.manual_seed(args.seed)
    if gpu: torch.cuda.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    experiment = f"gpt_{args.dataset}"
    save_folder = os.path.join(args.output_dir, experiment)
    os.makedirs(save_folder, exist_ok=True)
    t_writer = SummaryWriter(os.path.join(save_folder, 'train'), flush_secs=5)

    train_caps = json.load(open(os.path.join(DATA_PREFIX, "captions_train2017.json")))
    val_caps = json.load(open(os.path.join(DATA_PREFIX, "captions_val2017.json")))

    train_data = COCOCaption(train_caps)
    val_data = COCOCaption(val_caps)
    train_loader = DataLoader(train_data,
                              batch_size=args.train_batch_size,
                              drop_last=False,
                              num_workers=args.workers,
                              shuffle=True)
    val_loader = DataLoader(val_data,
                            batch_size=args.eval_batch_size,
                            drop_last=False,
                            num_workers=args.workers,
                            shuffle=False)
    total_step = len(train_loader) * args.num_train_epochs

    logging_file = f"gpt_{args.dataset}_{now.month}.{now.day}.log"
    logging = Logger(os.path.join(save_folder, logging_file))
    logging.info('\n*******************************************************************************\n')
    logging.info("the configuration:")
    logging.info(str(args).replace(',', '\n'))
    logging.info('Loading models...')


    model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir=CACHE_DIR)
    if args.load is not None:
        state = torch.load(os.path.join(args.output_dir, args.load, "model_best_val.pt"))
        model.load_state_dict(state)
        del state
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.2 * total_step,
                                                num_training_steps=total_step)

    logging.info("Begin training steps")
    logging.info("Total step: %d" % total_step)
    e = 0  # number of epoch
    num_iters = 0
    best_loss = 9e9
    et = 0
    optimizer.zero_grad()
    scaler = torch.cuda.amp.GradScaler()
    while (num_iters < total_step) and (et < args.early_stop):
        # Run epoch Training
        print('Training loop. Batches:', len(train_loader))
        logging.info('\n===========================================================================')
        logging.info("Training loop.       Batches: %d" % len(train_loader))

        ## record for every epoch losses
        losses = []
        for i, data_dict in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                encoded = tokenizer(data_dict, padding=True, return_tensors="pt")
                inputs = {
                    "input_ids": encoded['input_ids'].to(device),
                    "attention_mask": encoded['attention_mask'].to(device),
                    "labels": encoded["input_ids"].to(device)
                }
                outputs = model(**inputs)
                loss = outputs[0]
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())

            lr = scheduler.get_last_lr()[0]
            t_writer.add_scalar('loss', loss, num_iters)
            t_writer.add_scalar('loss', loss, num_iters)
            t_writer.add_scalar('lr', lr, num_iters)

            num_iters += 1

            if num_iters % args.test_iter == 0:
                eval_loss = evaluate(args, val_loader, model, tokenizer, logging, device)
                model.train()
                curr_loss = eval_loss
                if curr_loss <= best_loss:
                    best_loss = curr_loss
                    torch.save(model.state_dict(), os.path.join(save_folder, "model_best_val.pt"))
                    et = 0
                else:
                    et += 1

        e += 1
        logging.info(f"Finish Training for {e} Epochs..")

        if e % args.log_epoch == 0:
            logging.info("========================= Training Status =========================")
            logging.info("Epoch                : {}".format(e))
            logging.info("Avg. Loss            : {:.4f}".format(loss.item()))

        if et >= args.early_stop:
            logging.info("Early Stopping..")
            break
        if (args.num_train_epochs is not None) and (e >= args.num_train_epochs):
            break

if __name__ == '__main__':
    # args = parser.parse_args()
    args = parser.parse_args('--load gpt_coco_0 --per_gpu_train_batch_size 15'.split())
    run(args)


