from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torch.optim import Adam
import torch.nn as nn
import torch
import numpy as np
import os
# from seqeval.metrics.sequence_labeling import get_metrices
from config import Args
from config import Args
from model import BertForSC
from dataset import BertDataset
from transformers import BertTokenizer, ErnieForMaskedLM
import json
from trainer import Trainer

tokenizer = BertTokenizer.from_pretrained(Args.bert_dir)
class InputFeature:
    def __init__(self,
                 input_ids,
                 attention_mask,
                 token_type_ids,
                 seq_label_ids,):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.seq_label_ids = seq_label_ids


if Args.do_train:
    features=[]
    with open('./data/train.json', 'r') as train_data_file:
        for line in train_data_file:
            single_data=json.loads(line)
            seq_label_ids = 1 if single_data['label']=='positive' else 0
            inputs = tokenizer.encode_plus(
                text=single_data['content'],
                max_length=Args.max_len,
                padding='max_length',
                truncation='only_first',
                return_attention_mask=True,
                return_token_type_ids=True,
            )
            input_ids = torch.tensor(inputs['input_ids'], requires_grad=False)
            attention_mask = torch.tensor(inputs['attention_mask'], requires_grad=False)
            token_type_ids = torch.tensor(inputs['token_type_ids'], requires_grad=False)
            seq_label_ids = torch.tensor(seq_label_ids, requires_grad=False)
            features.append(InputFeature(input_ids,attention_mask,token_type_ids,seq_label_ids))
        train_dataset = BertDataset(features)
        train_loader = DataLoader(train_dataset, batch_size=Args.batchsize, shuffle=True)
    test_features=[]
    with open('./data/test.json', 'r') as test_data_file:
        for line in test_data_file:
            single_data=json.loads(line)
            seq_label_ids = 1 if single_data['label']=='positive' else 0
            inputs = tokenizer.encode_plus(
                text=single_data['content'],
                max_length=Args.max_len,
                padding='max_length',
                truncation='only_first',
                return_attention_mask=True,
                return_token_type_ids=True,
            )
            input_ids = torch.tensor(inputs['input_ids'], requires_grad=False)
            attention_mask = torch.tensor(inputs['attention_mask'], requires_grad=False)
            token_type_ids = torch.tensor(inputs['token_type_ids'], requires_grad=False)
            seq_label_ids = torch.tensor(seq_label_ids, requires_grad=False)
            test_features.append(InputFeature(input_ids,attention_mask,token_type_ids,seq_label_ids))
        test_dataset = BertDataset(test_features)
        test_loader = DataLoader(test_dataset, batch_size=Args.batchsize, shuffle=True)
    model = BertForSC(Args).to(Args.device)
    trainer=Trainer(model, Args)
    for i in range(Args.epoch):
        trainer.train(train_loader, test_loader, i)
        # trainer.test(test_loader, i)

# if __name__ = '__main__':
