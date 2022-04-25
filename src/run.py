
import os
import random

import torch

import torch.optim as optim
import numpy as np

from dataloader import get_data_loader
import evaluation
import warnings
warnings.filterwarnings("ignore")

from c2vRNNModel import c2vRNNModel
from config import Config

def setup_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    
    config = Config()

    setup_seed(0)

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        

    performance_list = []
    scores_list = []
    first_scores_list = []
    first_total_scores_list = []

    for fold in range(10):
        print("----",fold,"-th run----")
        train_loader, test_loader = get_data_loader(config.bs, config.questions, config.length, fold)
        node_count, path_count = np.load("np_counts.npy")

        model = c2vRNNModel(config.questions * 2,
                            config.hidden,
                            config.layers,
                            config.questions,
                            node_count, path_count, device) 

        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        loss_func = evaluation.lossFunc(config.questions, config.length, device)
        for epoch in range(config.epochs):
            print('epoch: ' + str(epoch))
            model, optimizer = evaluation.train_epoch(model, train_loader, optimizer,
                                              loss_func, config, device)
        first_total_scores, first_scores, scores, performance = evaluation.test_epoch(
            model, test_loader, loss_func, device, epoch, config, fold)
        first_total_scores_list.append(first_total_scores)
        scores_list.append(scores)
        first_scores_list.append(first_scores)
        performance_list.append(performance)
    print("Average scores of the first attempts:", np.mean(first_total_scores_list,axis=0))
    print("Average scores of all attempts:", np.mean(performance_list,axis=0))

if __name__ == '__main__':
    main()
