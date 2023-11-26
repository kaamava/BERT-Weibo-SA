import torch
from transformers import BertTokenizer, ErnieForMaskedLM
from transformers import BertModel
from config import Args
from model import BertForSC
import torch.nn.functional as F
tokenizer = BertTokenizer.from_pretrained(Args.bert_dir)
import pandas as pd
from tqdm import tqdm
def predict(text):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer.encode_plus(
            text=text,
            max_length=Args.max_len,
            padding='max_length',
            truncation='only_first',
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )
        for key in inputs.keys():
            inputs[key] = inputs[key]
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        seq_output = model(
            input_ids,
            attention_mask,
            token_type_ids,
        )
    return seq_output.numpy().reshape(2)

if __name__ == '__main__':
    model=BertForSC(Args)
    model.load_state_dict(torch.load(Args.load_dir, map_location='cpu'))
    df=pd.read_excel('微博111.xlsx')
    score_list=[]
    for text in tqdm(df['标题']):
        prediction=predict(text)
        if prediction[0]>=prediction[1]:
            score_list.append(0.5-prediction[0])
        else:
            score_list.append(prediction[1]-0.5)
    new_column_name='scores'
    df[new_column_name]=score_list
    df.to_excel('微博111.xlsx', index=False)