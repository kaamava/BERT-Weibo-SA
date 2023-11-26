import re
import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset

class BertDataset(Dataset):
    def __init__(self, features):
        self.features = features
        self.nums = len(self.features)

    def __len__(self):
        return self.nums

    def __getitem__(self, item):
        data = {
            'input_ids': self.features[item].input_ids.long(),
            'attention_mask': self.features[item].attention_mask.long(),
            'token_type_ids': self.features[item].token_type_ids.long(),
            'seq_label_ids': self.features[item].seq_label_ids.long(),
        }
        return data
    
if __name__ == '__main__':
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
    inputfeature=InputFeature(torch.randint(1,9, (512,)),torch.randint(1,9, (1,512)),torch.randint(1,9, (1,512)),torch.tensor([1]))
    Features=[]
    Features.append(inputfeature)
    Dataset=BertDataset(Features)
    print(Dataset[1]['input_ids'])