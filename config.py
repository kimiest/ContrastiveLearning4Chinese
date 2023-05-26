import torch

class Config():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_path = r'E:\Code\ContrastiveLearning_SimCSE\data\ocnli_sup_train_data.csv'
    dev_path = r'E:\Code\ContrastiveLearning_SimCSE\data\STS-B\dev.txt'
    test_path = r'E:\Code\ContrastiveLearning_SimCSE\data\STS-B\test.txt'

    plm_path = 'bert-base-chinese'
    tokenizer_path = 'bert-base-chinese'
    train_mode = 'supervise'  # train_mode should in ['supervise', 'unsupervise']
    pooler = 'cls'  # pooler should in ["cls", "pooler", "last-avg", "first-last-avg"]

    epochs = 5
    batch_size = 8
    eval_step = 100
    max_length = 25
    learning_rate = 2e-5
    dropout = 0.3


