import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import argparse
from torch.utils.data import DataLoader
import random

from dataset.CSIG_dataset import AudioDataset
from network.model import AudioModel, AudioModelV2, AudioModelV3
from utils import Logger, AverageMeter, calculate_metrics, LabelSmoothing


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_model(model, criterion, optimizer, epoch):
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    training_process = tqdm(train_loader)
    for i, (XI, label) in enumerate(training_process):
        if i > 0:
            training_process.set_description(
                "Epoch: %d, Loss: %.4f, Acc: %.4f" % (epoch, losses.avg.item(), accuracies.avg.item()))

        x = Variable(XI.cuda(device_id))
        label = Variable(label.cuda(device_id))
        # label = Variable(torch.LongTensor(label).cuda(device_id))
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)
        # Compute and print loss
        loss = criterion(y_pred, label)
        # print(nn.Softmax(dim=1)(y_pred), y_pred)
        acc = calculate_metrics(nn.Softmax(dim=1)(y_pred).cpu(), label.cpu())
        losses.update(loss.cpu(), x.size(0))
        accuracies.update(acc, x.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()
    train_logger.log(phase="train", values={
        'epoch': epoch,
        'loss': format(losses.avg.item(), '.4f'),
        'acc': format(accuracies.avg.item(), '.4f'),
        'lr': optimizer.param_groups[0]['lr']
    })
    print("Train:\t Loss:{0:.4f} \t Acc:{1:.4f}".format(losses.avg, accuracies.avg))


def eval_model(model, epoch, eval_loader, is_save=True):
    eval_criterion = nn.CrossEntropyLoss()
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    eval_process = tqdm(eval_loader)
    with torch.no_grad():
        for i, (img, label) in enumerate(eval_process):
            if i > 0:
                eval_process.set_description("Epoch: %d, Loss: %.4f, Acc: %.4f" %
                                             (epoch, losses.avg.item(), accuracies.avg.item()))
            img, label = Variable(img.cuda(device_id)), Variable(label.cuda(device_id))
            y_pred = model(img)

            loss = eval_criterion(y_pred, label)
            acc = calculate_metrics(nn.Softmax(dim=1)(y_pred).cpu(), label.cpu())

            losses.update(loss.cpu(), img.size(0))
            accuracies.update(acc, img.size(0))
    if is_save:
        train_logger.log(phase="val", values={
            'epoch': epoch,
            'loss': format(losses.avg.item(), '.4f'),
            'acc': format(accuracies.avg.item(), '.4f'),
            'lr': optimizer.param_groups[0]['lr']
        })
    print("Val:\t Loss:{0:.4f} \t Acc:{1:.4f}".format(losses.avg, accuracies.avg))
    return accuracies.avg


parser = argparse.ArgumentParser(description="CSIG audio  @cby Training")
parser.add_argument(
    "--device_id", default=0, help="Setting the GPU id", type=int
)
parser.add_argument(
    "--k", default=-1, help="The value of K Fold", type=int
)
parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                    nargs=argparse.REMAINDER)
args = parser.parse_args()

if __name__ == "__main__":
    # 设置随机数种子
    setup_seed(20)
    # set dataset
    train_path = '/pubdata/chenby/dataset/CSIG/FMFCC_Audio_train/'
    k = args.k
    batch_size = 64
    test_batch_size = 128
    is_one_hot = False
    epoch_start = 0
    num_epochs = 10
    device_id = args.device_id
    lr = 1e-3

    model_name = 'efficientnet-b2'  # efficientnet-b0
    # writeFile = './output/logs/vggish_' + model_name + '_LS'
    # store_name = './output/weights/vggish_' + model_name + '_LS'
    writeFile = './output/logs/' + model_name + '_96_hop48'  # 有重叠
    store_name = './output/weights/' + model_name + '_96_hop48'
    if k != -1:
        writeFile += f'_k{k}'
        store_name += f'_k{k}'
    if store_name and not os.path.exists(store_name):
        os.makedirs(store_name)

    # audio_model = AudioModel(model_name=model_name)
    audio_model = AudioModelV3(model_name=model_name)
    model_path = None
    # model_path = '/pubdata/chenby/weights/vgg_16_v4/audio9_acc0.9988.pth'
    if model_path is not None:
        audio_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print('Model found in {}'.format(model_path))
    else:
        print('No network found, initializing random network.')

    audio_model = audio_model.cuda(device_id)

    criterion = nn.CrossEntropyLoss()
    # criterion = LabelSmoothing(smoothing=0.05).cuda(device_id)

    is_training = True
    if is_training:
        dataset = AudioDataset(root_path=train_path, is_one_hot=is_one_hot, k=k)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        eval_dataset = AudioDataset(root_path=train_path, data_type='Val', k=k)
        eval_loader = DataLoader(eval_dataset, batch_size=test_batch_size, shuffle=False)

        optimizer = optim.SGD(audio_model.parameters(), lr=lr, momentum=0.9)  # 原始使用
        # optimizer = optim.AdamW(audio_model.parameters(), lr=lr, weight_decay=4e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
        train_logger = Logger(model_name=writeFile, header=['epoch', 'loss', 'acc', 'lr'])
        best_acc = 0.5 if epoch_start == 0 else eval_model(audio_model, epoch_start - 1, eval_loader, is_save=False)
        for epoch in range(epoch_start, num_epochs):
            train_model(audio_model, criterion, optimizer, epoch)
            acc = eval_model(audio_model, epoch, eval_loader)
            if best_acc < acc:
                best_acc = acc
                torch.save(audio_model.state_dict(), '{}/{}_acc{:.4f}.pth'.format(store_name, 'audio' + str(epoch), acc))
            print('current best acc:', best_acc)
    else:
        eval_dataset = AudioDataset(root_path=train_path, data_type='Val', k=k)
        eval_loader = DataLoader(eval_dataset, batch_size=test_batch_size, shuffle=False)
        acc = eval_model(audio_model, -1, eval_loader, is_save=False)
