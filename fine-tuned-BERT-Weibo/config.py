class Args:
    bert_dir='./chinese-bert-wwm-ext'
    lr=1e-5
    epoch=9
    device='cuda:6'
    max_len=512
    batchsize=32
    do_train=True
    hidden_dropout_prob=0.1
    hidden_size=768
    seq_num_labels=2
    save_path='./model/softmax'
    load_dir='./model/softmax/epoch4step50loss0.387990266084671'
