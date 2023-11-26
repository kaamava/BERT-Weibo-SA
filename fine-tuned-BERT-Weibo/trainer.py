from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torch.optim import Adam
import torch.nn as nn
import torch
import numpy as np
import os
# from seqeval.metrics.sequence_labeling import get_entities
from tqdm import tqdm


class Trainer:
    def __init__(self, model, Args):
        self.model=model
        self.Args=Args
        self.criterion=nn.CrossEntropyLoss()
        self.optimizer=Adam(self.model.parameters(), lr=Args.lr)
        self.epoch=Args.epoch
        self.device=Args.device
        self.save_path=Args.save_path

    def train(self, train_loader, test_loader, epoch_i):
        self.model.train()
        for step, train_batch in tqdm(enumerate(train_loader)):
            for key in train_batch.keys():
                train_batch[key] = train_batch[key].to(self.device)
            input_ids = train_batch['input_ids']
            attention_mask = train_batch['attention_mask']
            token_type_ids = train_batch['token_type_ids']
            seq_label_ids = train_batch['seq_label_ids']
            seq_output= self.model(
                input_ids,
                attention_mask,
                token_type_ids,
            )
            #token_output维度[batch_size, sequence_len, token_num_labels]
            #seq_output维度[batch_size,1]
            # 选出不是padding的logits和label
            seq_loss = self.criterion(seq_output, seq_label_ids)
            self.model.zero_grad()
            seq_loss.backward()
            self.optimizer.step()
            # print(f'[train] epoch:{epoch_i+1} loss:{seq_loss.item()}')
            if (step+1)%50==0:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'epoch{}step{}loss{}'.format(epoch_i+1,step+1,seq_loss.item() )))
                self.test(test_loader, step, epoch_i)

    def test(self, test_loader, step_i, epoch_i):
        self.model.eval()
        seq_preds = []
        seq_trues = []
        with torch.no_grad():
            for step, test_batch in tqdm(enumerate(test_loader)):
                for key in test_batch.keys():
                    test_batch[key] = test_batch[key].to(self.device)
                input_ids = test_batch['input_ids']
                attention_mask = test_batch['attention_mask']
                token_type_ids = test_batch['token_type_ids']
                seq_label_ids = test_batch['seq_label_ids']
                seq_output = self.model(
                    input_ids,
                    attention_mask,
                    token_type_ids,
                )
                seq_output = seq_output.detach().cpu().numpy()
                seq_output = np.argmax(seq_output, -1)
                seq_label_ids = seq_label_ids.detach().cpu().numpy()
                seq_label_ids = seq_label_ids.reshape(-1)
                seq_preds.extend(seq_output)
                seq_trues.extend(seq_label_ids)
        acc, precision, recall, f1 = self.get_metrices(seq_trues, seq_preds, 'cls')
        with open('result.txt', 'a') as result:
            result.write('epoch:{}step;{}：\naccuracy:{}\nprecision:{}\nrecall:{}\nf1:{}\n'.format(epoch_i+1, step_i+1,acc, precision, recall, f1))

    def get_metrices(self, trues, preds, mode):
        if mode == 'cls':
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            acc = accuracy_score(trues, preds)
            precision = precision_score(trues, preds, average='micro')
            recall = recall_score(trues, preds, average='micro')
            f1 = f1_score(trues, preds, average='micro')
        elif mode == 'ner':
            from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score
            acc = accuracy_score(trues, preds)
            precision = precision_score(trues, preds)
            recall = recall_score(trues, preds)
            f1 = f1_score(trues, preds)
        return acc, precision, recall, f1