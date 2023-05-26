# Python库
from tqdm import tqdm
from loguru import logger
import os
import time
import json

# numpy,pandas scipy库
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# torch和transformers库
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wandb
from transformers import BertModel, BertConfig, BertTokenizer

# 自己的
from model import SimcseModel, unsup_loss, sup_loss
from utils.get_dataloader import get_train_dev_dataloader
from utils.others import seed_everything
from config import Config


def train(model, optimizer, train_dl, dev_dl, Config, run):
    logger.info("开始训练")
    model.train()
    best = 0
    for epoch in range(Config.epochs):
        logger.info('start {}-th epoch training'.format(epoch + 1))
        for batch_idx, batch in enumerate(tqdm(train_dl)):
            step = epoch * len(train_dl) + batch_idx + 1
            # [batch, n, seq_len] -> [batch * n, sql_len]
            sql_len = batch.shape[-1]
            batch_inputs = batch.view(-1, sql_len).to(Config.device)

            out = model(input_ids=batch_inputs)
            if Config.train_mode == 'unsupervise':
                loss = unsup_loss(out, Config.device)
            else:
                loss = sup_loss(out, Config.device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % Config.eval_step == 0:
                corrcoef = evaluate(model, dev_dl, Config.device)
                logger.info(
                    'training loss:{}, corrcoef: {} in step {} epoch {}'.format(
                        loss, corrcoef, step, epoch+1
                    )
                )
                run.log({'step': step, 'training loss:': loss, 'corrcoef': corrcoef})
                model.train()
                if  corrcoef > best:
                    best = corrcoef
                    torch.save(model.state_dict(), 'results/checkpoint.pt')
                    logger.info('higher corrcoef: {} in step {} epoch {}, save model'.format(best, step, epoch+1))


def evaluate(model, dataloader, device):
    model.eval()
    sim_tensor = torch.tensor([], device=device)
    label_array = np.array([])
    with torch.no_grad():
        for source, target, label in dataloader:
            # source
            source_inputs = source.squeeze(1).to(device)  # [batch, 1, seq_len] -> [batch, seq_len]
            source_preds = model(input_ids=source_inputs)
            # target
            target_inputs = target.squeeze(1).to(device)  # [batch, 1, seq_len] -> [batch, seq_len]
            target_preds = model(input_ids=target_inputs)
            # concat
            sim = F.cosine_similarity(source_preds, target_preds, dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)
            label_array = np.append(label_array, np.array(label))
    # corrcoef
    return spearmanr(label_array, sim_tensor.cpu().numpy()).correlation


if __name__ == '__main__':
    seed_everything(2023)

    # 加载模型和优化器
    model = SimcseModel(pretrained_model=Config.plm_path,
                        pooling=Config.pooler,
                        dropout=Config.dropout).to(Config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate)

    # 加载train和dev数据
    train_dl, dev_dl = get_train_dev_dataloader(Config)

    # 训练
    run = wandb.init(project='project', name='test001')
    run.watch(model)
    train(model, optimizer, train_dl, dev_dl, Config, run)

