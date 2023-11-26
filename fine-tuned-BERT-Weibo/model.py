import torch.nn as nn
from transformers import BertTokenizer, ErnieForMaskedLM
from transformers import BertModel
import torch.nn.functional as F
from config import Args
class BertForSC(nn.Module):
    def __init__(self, config):
        super(BertForSC, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(Args.bert_dir)
        self.bert_config = self.bert.config
        self.SC = nn.Sequential(
            nn.Dropout(Args.hidden_dropout_prob),
            #hidden_size是768，这个和bert模型的最后一层是一致的
            nn.Linear(config.hidden_size, config.seq_num_labels),
        )
    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                ):
        bert_output = self.bert(input_ids, attention_mask, token_type_ids)
        pooler_output = bert_output[1]
        #(batch_size, sequence_length, hidden_size)
        seq_output = self.SC(pooler_output)
        seq_output=F.softmax(seq_output, dim=1)
        return seq_output