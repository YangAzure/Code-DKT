import tqdm
import torch
import logging
import seaborn as sns
import numpy as np
import matplotlib.pylab as plt
import torch.nn as nn
from sklearn import metrics

def performance_granular(batch, pred, ground_truth, prediction, epoch, config):
    
    
    preds = {k:[] for k in range(config.questions)}
    gts = {k:[] for k in range(config.questions)}
    first_preds = {k:[] for k in range(config.questions)}
    first_gts = {k:[] for k in range(config.questions)}
    scores = {}
    first_scores = {}

    
    for s in range(pred.shape[0]):
        delta = (batch[s][1:, 0:config.questions] + batch[s][1:, config.questions:config.questions*2])
        temp = pred[s][:config.length-1].mm(delta.T)
        index = torch.tensor([[i for i in range(config.length-1)]],
                             dtype=torch.long)
        p = temp.gather(0, index)[0].detach().cpu().numpy()
        a = (((batch[s][:, 0:config.questions] - batch[s][:, config.questions:config.questions*2]).sum(1) + 1) // 2)[1:].detach().cpu().numpy()

        for i in range(len(p)):
            if p[i] > 0:
                p = p[i:]
                a = a[i:]
                delta = delta.detach().cpu().numpy()[i:]
                break
        
        
        for i in range(len(p)):
            for j in range(config.questions):
                if delta[i,j] == 1:
                    preds[j].append(p[i])
                    gts[j].append(a[i])
                    if i == 0 or delta[i-1,j] != 1:
                        first_preds[j].append(p[i])
                        first_gts[j].append(a[i])
                        
    first_total_gts = []
    first_total_preds = []
    for j in range(config.questions):
        f1 = metrics.f1_score(gts[j], np.where(np.array(preds[j])>0.5, 1, 0))
        recall = metrics.recall_score(gts[j], np.where(np.array(preds[j])>0.5, 1, 0))
        precision = metrics.precision_score(gts[j], np.where(np.array(preds[j])>0.5, 1, 0))
        acc = metrics.accuracy_score(gts[j], np.where(np.array(preds[j])>0.5, 1, 0))
        try:
            auc = metrics.roc_auc_score(gts[j], preds[j])
        except ValueError:
            auc = 0.5
        scores[j]=[auc,f1,recall,precision,acc]
        print('problem '+str(j)+' auc: ' + str(auc) + ' f1: ' + str(f1) + ' recall: ' + str(recall) +
          ' precision: ' + str(precision) + ' acc: ' +
                str(acc))
        
        
        first_f1 = metrics.f1_score(first_gts[j], np.where(np.array(first_preds[j])>0.5, 1, 0))
        first_recall = metrics.recall_score(first_gts[j], np.where(np.array(first_preds[j])>0.5, 1, 0))
        first_precision = metrics.precision_score(first_gts[j], np.where(np.array(first_preds[j])>0.5, 1, 0))
        first_acc = metrics.accuracy_score(first_gts[j], np.where(np.array(first_preds[j])>0.5, 1, 0))
        try:
            first_auc = metrics.roc_auc_score(first_gts[j], first_preds[j])
        except ValueError:
            first_auc = 0.5
            
        first_total_gts.extend(first_gts[j])
        first_total_preds.extend(first_preds[j])
        
        first_scores[j]=[first_auc,first_f1,first_recall,first_precision,first_acc]
        print('First prediction for problem '+str(j)+' auc: ' + str(first_auc) + ' f1: ' + str(first_f1) + ' recall: ' + str(first_recall) + ' precision: ' + str(first_precision) + ' acc: ' + str(first_acc))
    
    f1 = metrics.f1_score(ground_truth.detach().numpy(),
                          torch.where(prediction>0.5,torch.tensor(1), torch.tensor(0)).detach().numpy())
    recall = metrics.recall_score(ground_truth.detach().numpy(),
                                  torch.where(prediction>0.5,torch.tensor(1), torch.tensor(0)).detach().numpy())
    precision = metrics.precision_score(
        ground_truth.detach().numpy(),
        torch.where(prediction>0.5,torch.tensor(1), torch.tensor(0)).detach().numpy())
    acc = metrics.accuracy_score(
        ground_truth.detach().numpy(),
        torch.where(prediction>0.5,torch.tensor(1), torch.tensor(0)).detach().numpy())
    auc = metrics.roc_auc_score(
        ground_truth.detach().numpy(),
        prediction.detach().numpy())
    print('auc: ' + str(auc) + ' f1: ' + str(f1) + ' recall: ' + str(recall) +
          ' precision: ' + str(precision) + ' acc: ' +
                str(acc))
    
    
    
    first_total_f1 = metrics.f1_score(first_total_gts, np.where(np.array(first_total_preds)>0.5, 1, 0))
    first_total_recall = metrics.recall_score(first_total_gts, np.where(np.array(first_total_preds)>0.5, 1, 0))
    first_total_precision = metrics.precision_score(first_total_gts, np.where(np.array(first_total_preds)>0.5, 1, 0))
    first_total_acc = metrics.accuracy_score(first_total_gts, np.where(np.array(first_total_preds)>0.5, 1, 0))
    try:
        first_total_auc = metrics.roc_auc_score(first_total_gts, first_total_preds)
    except ValueError:
        first_total_auc = 0.5
    
    first_total_scores = [first_total_auc,first_total_f1,first_total_recall,first_total_precision,first_total_acc]
    
    return first_total_scores, first_scores, scores, [auc,f1,recall,precision,acc]

def plot_heatmap(batch, pred, fold, batch_n, config):
    
    # TODO: No hardcoding problem dict but what about other assignments?
    problem_dict = {"000000010":"1",
                    "000000001":"3",
                    "000010000":"5",
                    "010000000":"13",
                    "001000000":"232",
                    "000100000":"233",
                    "100000000":"234",
                    "000001000":"235",
                    "000000100":"236"
                   }
    problems = []
    for s in range(pred.shape[0]):
        
        delta = (batch[s][1:, 0:config.questions] + batch[s][1:, config.questions:config.questions*2]).detach().cpu().numpy()
        
        a = (((batch[s][:, 0:config.questions] - batch[s][:, config.questions:config.questions*2]) + 1) // 2)[1:].detach().cpu().numpy()
        p = pred[s].detach().cpu().numpy()

        for i in range(len(delta)):
            if np.sum(delta, axis=1)[i] > 0:
                p = p[i:]
                a = a[i:]
                delta = delta[i:]
                break
        
        problems = [problem_dict["".join([str(int(i)) for i in sk])] for sk in delta]
        
        plt.figure(figsize=(15, 6), dpi=80)
    
        ax = sns.heatmap(p.T, annot=a.T, linewidth=0.5, vmin=0, vmax=1, cmap="Blues")

        plt.xticks(np.arange(len(problems))+0.5, problems, rotation=45)
        plt.yticks(np.arange(config.questions)+0.5, ['234', '13', '232', '233', '5', '235', '236', '1', '3'], rotation=45)
        plt.xlabel("Attempting Problem")
        plt.ylabel("Problem")

        
        plt.title("Heatmap for student "+str(s)+" fold "+str(fold))
        plt.tight_layout()
        plt.savefig("heatmaps/b"+str(batch_n)+"_s"+str(s)+"_f"+str(fold)+".png")
        


class lossFunc(nn.Module):
    def __init__(self, num_of_questions, max_step, device):
        super(lossFunc, self).__init__()
        self.crossEntropy = nn.BCELoss()
        self.num_of_questions = num_of_questions
        self.max_step = max_step
        self.device = device


    def forward(self, pred, batch):
        loss = 0
        prediction = torch.tensor([])
        ground_truth = torch.tensor([])
        pred = pred.to('cpu')

        for student in range(pred.shape[0]):

            delta = batch[student][:, 0:self.num_of_questions] + batch[
                student][:, self.num_of_questions:self.num_of_questions*2]  # shape: [length, questions]
            temp = pred[student][:self.max_step-1].mm(delta[1:].t())
            index = torch.tensor([[i for i in range(self.max_step-1)]],
                                 dtype=torch.long)
            p = temp.gather(0, index)[0]
            a = (((batch[student][:, 0:self.num_of_questions] -
                   batch[student][:, self.num_of_questions:self.num_of_questions*2]).sum(1) + 1) //
                 2)[1:]
            
            for i in range(len(p)):
                if p[i] > 0:
                    p = p[i:]
                    a = a[i:]
                    break


            loss += self.crossEntropy(p, a)
            prediction = torch.cat([prediction, p])
            ground_truth = torch.cat([ground_truth, a])

        return loss, prediction, ground_truth
    
    

def train_epoch(model, trainLoader, optimizer, loss_func, config, device):
    model.to(device)
    for batch in tqdm.tqdm(trainLoader, desc='Training:    ', mininterval=2):
        batch_new = batch[:,:-1,:].to(device)
        pred = model(batch_new)
        loss, prediction, ground_truth = loss_func(pred, batch[:,:,:config.questions*2])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, optimizer


def test_epoch(model, testLoader, loss_func, device, epoch, config, fold):
    model.to(device)
    ground_truth = torch.tensor([])
    prediction = torch.tensor([])
    full_data = torch.tensor([])
    preds = torch.tensor([])
    batch_n = 0
    for batch in tqdm.tqdm(testLoader, desc='Testing:     ', mininterval=2):
        batch_new = batch[:,:-1,:].to(device)
        pred = model(batch_new)
        loss, p, a = loss_func(pred, batch)
        
        prediction = torch.cat([prediction, p])
        ground_truth = torch.cat([ground_truth, a])
        full_data = torch.cat([full_data, batch])
        preds = torch.cat([preds, pred.cpu()])
        # plot_heatmap(batch, pred, fold, batch_n)
        batch_n += 1

    return performance_granular(full_data, preds, ground_truth, prediction, epoch, config)


