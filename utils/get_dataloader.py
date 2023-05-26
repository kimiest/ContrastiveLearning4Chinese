from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from loguru import logger
import pandas as pd


def read_data(mode, path, train_mode='unsupervise'):
    '''
    从硬盘读取数据
    '''
    assert mode in ['train', 'eval'], 'mode应设置为："train"或者"eval"'
    assert train_mode in ['supervise', 'unsupervise'], 'train_mode应设置为："supervise"或者"unsupervise"'
    if mode == 'train':
        if train_mode == 'unsupervise':
            data = []
            with open(path, encoding='utf-8') as f:
                for line in f.readlines():
                    data.append(line)
            return data  # data=[sentence, sentence,...]
        elif train_mode == 'supervise':
            data_df = pd.read_csv(path)
            data = data_df.values.tolist()  # data=[[anchor, entailment, contradiction],[],...]
            return data

    elif mode == 'eval':  # dev数据和test数据的读取格式一致，因此统称为读取eval数据
        data = []
        with open(path, encoding='utf8') as f:
            lines = f.readlines()
        for line in lines:
            _, sent1, sent2, score = line.strip().split('||')
            data.append((sent1, sent2, score))
        return data  # [(sent1, sent2, score),...]


class TrainDataset(Dataset):
    '''
    训练数据数值化
    '''
    def __init__(self, data, Config):
        tokenizer = AutoTokenizer.from_pretrained(Config.tokenizer_path)
        if Config.train_mode == 'unsupervise':  # data=[sentence, sentence,...]
            features = tokenizer(data, max_length=Config.max_length, truncation=True, padding='max_length',
                                 return_tensors='pt')['input_ids']  # features.shape=[num_examples, seq_len]
            self.inputs = features.unsqueeze(1).repeat(1, 2, 1)     # self.inputs.shape=[num_examples, 2, seq_len]
        elif Config.train_mode == 'supervise':  # data=[[anchor, entailment, contradiction],[]...]
            self.inputs = []
            for x in data:
                one_input = tokenizer(x, max_length=Config.max_length, truncation=True, padding='max_length',
                                      return_tensors='pt')['input_ids']  # one_input.shape=[3, 200]
                self.inputs.append(one_input)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index]


class EvalDataset(Dataset):
    '''
    开发验证/测试数据数值化
    '''
    def __init__(self, data, Config):  # data=[(sent1, sent2, score),...]
        self.tokenizer = AutoTokenizer.from_pretrained(Config.tokenizer_path)
        self.data = data
        self.Config = Config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sent1 = self.tokenizer(self.data[index][0], max_length=self.Config.max_length, truncation=True,
                               padding='max_length', return_tensors='pt')['input_ids']
        sent2 = self.tokenizer(self.data[index][1], max_length=self.Config.max_length, truncation=True,
                               padding='max_length', return_tensors='pt')['input_ids']
        score = float(self.data[index][2])  # sent1.shape=[batch, 1, seq_len]  score.shape=[batch]
        return sent1, sent2, score


def get_train_dev_dataloader(Config):
    '''返回train和dev数据'''
    train_data = read_data(mode='train', path=Config.train_path, train_mode=Config.train_mode)
    train_dl = DataLoader(TrainDataset(train_data, Config), batch_size=Config.batch_size, shuffle=True, pin_memory=True)
    dev_data = read_data(mode='eval', path=Config.dev_path)
    dev_dl = DataLoader(EvalDataset(dev_data, Config), batch_size=Config.batch_size, shuffle=True, pin_memory=True)
    return train_dl, dev_dl


def get_test_dataloader(Config):
    '''返回test数据'''
    logger.info('加载test数据')
    test_data = read_data(mode='eval', path=Config.test_path)
    test_dl = DataLoader(EvalDataset(test_data, Config), batch_size=Config.batch_size, shuffle=True, pin_memory=True)
    return test_dl


if __name__ == '__main__':
    from config import Config
    train_dl, dev_dl = get_train_dev_dataloader(Config)
    for batch in train_dl:
        print(f'train集一个batch的形状：{batch.shape}')
        break
    for batch in dev_dl:
        print(f'dev集中sent1形状：{batch[0].shape}，sent2形状：{batch[1].shape}，score形状：{batch[2].shape}')
        break
    test_dl = get_test_dataloader(Config)
    for batch in test_dl:
        print(f'test集中sent1形状：{batch[0].shape}，sent2形状：{batch[1].shape}，score形状：{batch[2].shape}')
        break
