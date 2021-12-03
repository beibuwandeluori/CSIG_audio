import tensorboardX
from sklearn.metrics import log_loss, accuracy_score, precision_score, average_precision_score, roc_auc_score, recall_score
import torch
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):
    def __init__(self, model_name, header):
        self.header = header
        self.writer = tensorboardX.SummaryWriter(model_name)

    def __del(self):
        self.writer.close()

    def log(self, phase, values):
        epoch = values['epoch']

        for col in self.header[1:]:
            self.writer.add_scalar(phase + "/" + col, float(values[col]), int(epoch))


def calculate_metrics(outputs, targets, metric_name='acc'):
    if len(targets.data.numpy().shape) > 1:
        _, targets = torch.max(targets.detach(), dim=1)
    if len(outputs.data.numpy().shape) > 1 and outputs.data.numpy().shape[1] == 1:  # 尾部是sigmoid
        outputs = torch.cat([1-outputs, outputs], dim=1)

    # print(outputs.shape, targets.shape, pred_labels.size())
    if metric_name == 'acc':
        pred_labels = torch.max(outputs, 1)[1]
        return accuracy_score(targets.data.numpy(), pred_labels.detach().numpy())
    elif metric_name == 'auc':
        pred_labels = outputs[:, 1]  # 为假的概率
        return roc_auc_score(targets.data.numpy(), pred_labels.detach().numpy())

