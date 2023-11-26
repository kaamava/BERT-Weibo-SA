import os
import json
def select_data(data_path, ratio):
    pos_data=[]
    neg_data=[]
    for file_name in os.listdir(data_path):
        try:
            with open(os.path.join(data_path, file_name), 'r') as file:
                data = json.load(file)
                for single_data in data:
                    if single_data['label']=='happy':
                        pos_data.append(single_data['content'])
                    elif single_data['label']=='angry' or single_data['label']=='sad' or single_data['label']=='fear':
                        neg_data.append(single_data['content'])
        except:
            pass
    pos_data = list(set(pos_data))
    neg_data = list(set(neg_data))
    category_len=min(len(pos_data), len(neg_data))
    with open(os.path.join(data_path, 'train.json'), 'w') as w_file:
        for i, _ in enumerate(pos_data[:round(category_len*ratio)]):
            w_file.write(json.dumps({'content': _, 'label': 'positive'}, ensure_ascii=False)+'\n')
        for i, _ in enumerate(neg_data[:round(category_len*ratio)]):
            w_file.write(json.dumps({'content': _, 'label': 'negtive'}, ensure_ascii=False)+'\n')
    with open(os.path.join(data_path, 'test.json'), 'w') as w_file:
        for i, _ in enumerate(pos_data[round(category_len*ratio):category_len]):
            w_file.write(json.dumps({'content': _, 'label': 'positive'}, ensure_ascii=False)+'\n')
        for i, _ in enumerate(neg_data[round(category_len*ratio):category_len]):
            w_file.write(json.dumps({'content': _, 'label': 'negtive'}, ensure_ascii=False)+'\n')
if __name__ == '__main__':
    select_data('./data', 0.8)