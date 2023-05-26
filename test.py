from loguru import logger
import torch

from model import SimcseModel
from utils.get_dataloader import get_test_dataloader
from train import evaluate
from config import Config


if __name__ == '__main__':
    logger.info('开始测试')
    model = SimcseModel(pretrained_model=Config.plm_path,
                        pooling=Config.pooler,
                        dropout=Config.dropout).to(Config.device)
    test_dl = get_test_dataloader(Config)
    model.load_state_dict(torch.load('results/checkpoint.pt'))
    model.eval()
    corrcoef = evaluate(model, test_dl, Config.device)
    logger.info('testset corrcoef:{}'.format(corrcoef))