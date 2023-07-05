# -*- coding:utf-8 -*-
import os
import torch
import argparse
import logging

import numpy as np
import tensorboard_logger as tb
import torch.nn.functional as F

from models import *
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datasets.cifar import load_data
from utils.tools import ModelEma,accuracy,evaluate,AverageMeter

class GalleryLoss(nn.Module):
    def __init__(self, num_classes=10, label_smoothing=0, consistency=0, class_balance=0):
        super(GalleryLoss, self).__init__()
        self.num_classes = num_classes
        self.consistency = consistency
        self.class_balance = class_balance
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing).cuda()

    def forward(self, outputs1, outputs2, labels, labels_mix = None, lam=1):
        y_hat1 = F.softmax(outputs1, dim=1)
        y_hat2 = F.log_softmax(outputs2, dim=1)
        # consistency loss
        loss_con = F.kl_div(y_hat2, y_hat1, reduction='batchmean')
        # class_balance loss
        avg_prediction = torch.mean(y_hat1, dim=0)
        prior_distr = 1.0 / self.num_classes * torch.ones_like(avg_prediction)
        avg_prediction = torch.clamp(avg_prediction, min=1e-5, max=1.)
        loss_class_balance = torch.mean(-(prior_distr * torch.log(avg_prediction)).sum(dim=0))
        # cross entropy with mixup
        if labels_mix is None:
            loss_ce = self.ce(outputs1, labels)
        else:
            loss_ce = lam*self.ce(outputs1, labels) + (1-lam)*self.ce(outputs1, labels_mix)

        # final loss
        loss = loss_ce + self.consistency*loss_con + self.class_balance*loss_class_balance
        return loss

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def train(model, ema_model, train_loader, criterion, optimizer, alpha=0):
    model.train()
    loss_meter = AverageMeter()
    for batch_idx, (images, labels, _, indexes) in enumerate(train_loader):
        images1, images2, labels, indexes = images[0].cuda(), images[1].cuda(), labels.long().cuda(), indexes.cuda()
        outputs1 = model(images1)
        outputs2 = model(images2)

        loss = criterion(outputs1, outputs2, labels)
        if alpha > 0:
            inputs, targets_a, targets_b, lam = mixup_data(images1, labels, alpha)
            inputs, targets_a, targets_b = map(Variable, (inputs,targets_a, targets_b))
            outputs1 = model(inputs)
            outputs2 = model(images2)

            loss = criterion(outputs1, outputs2, targets_a, targets_b, lam)

        else:
            outputs1 = model(images1)
            outputs2 = model(images2)
            loss = criterion(outputs1, outputs2, labels)

        loss_meter.update(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # update ema
        ema_model.update(model)

    train_acc = accuracy(model, train_loader)
    ema_train_acc = accuracy(ema_model.module, train_loader)
    return train_acc, ema_train_acc, loss_meter.avg

def main(args, logger, tb_logger, acc_list, ema_acc_list):
    human_noise_type_map = {'clean':'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1', 'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label', 'noisy100': 'noisy_label'}
    if args.noise_type in human_noise_type_map:
        args.noise_type = human_noise_type_map[args.noise_type]
    
    train_dataset, test_dataset, num_classes, num_examples = load_data(args, dataset=args.dataset,
                                                                       noisy_type=args.noise_type,
                                                                       noise_rate=args.noise_rate)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.net.lower() == 'resnet18':
        model = ResNet18(num_classes=num_classes).cuda()
    elif args.net.lower() == 'preresnet18':
        model = PreResNet18(num_classes=num_classes).cuda()
    else:
        raise NotImplementedError("Only support ResNet18 and PreResNet18")

    logger.info("build model done")
    
    ema_m = ModelEma(model, args.ema_decay) # 0.9997
    logger.info("build ema model done")

    # criterion = nn.CrossEntropyLoss().cuda()
    criterion = GalleryLoss(num_classes, args.label_smoothing, args.consistency, args.class_balance).cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=0.0002)
    
    best_acc = 0.0
    best_ema_acc = 0.0

    for epoch in tqdm(range(args.epoch)):
        train_acc, ema_train_acc, train_loss = train(model=model, ema_model=ema_m, train_loader=train_loader,criterion=criterion, optimizer=optimizer, alpha=args.alpha)
        
        tb_logger.log_value('train_loss', train_loss, epoch)
        tb_logger.log_value('train_acc', train_acc, epoch)
        tb_logger.log_value('ema_train_acc', ema_train_acc, epoch)

        scheduler.step()
        test_acc= evaluate(model, test_loader)
        ema_test_acc = evaluate(ema_m.module, test_loader)

        best_acc = max(test_acc, best_acc)
        best_ema_acc = max(ema_test_acc, best_ema_acc)

        print(f'Epoch[{epoch}/{args.epoch-1}]: Train Acc: {train_acc:.2f} Train EMA Acc: {ema_train_acc:.2f} Test Acc: {test_acc:.2f} Test EMA Acc:{ema_test_acc}')
        logger.info(f'Epoch[{epoch}/{args.epoch-1}]: Train Acc: {train_acc:.2f} Train EMA Acc: {ema_train_acc:.2f} Test Acc: {test_acc:.2f} Test EMA Acc:{ema_test_acc}')

    print(f"best acc is {best_acc}")
    logger.info(f"best acc is {best_acc}")
    print(f"EMA best acc is {best_ema_acc}")
    logger.info(f"EMA best acc is {best_ema_acc}")

    acc_list.append(best_acc)
    ema_acc_list.append(best_ema_acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str, help = ' cifar10 or cifar100', default = 'cifar10')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=2, help='how many subprocesses to use for data loading')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--lr', type = float, default = 0.02)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--net', type = str, choices=['ResNet18', 'PreResNet18'], default='ResNet18')
    parser.add_argument('--data_path', type = str, help='data path', default='./data')
    parser.add_argument('--log_path', type = str, help='save logs', default='./experiments/logs')
    parser.add_argument('--tb_log_path', type = str, help='save logs', default='./experiments/tb_logs')
    '''noise type
    sysmetric noise : {symmetric}
    asysmetric noise : {pairflip, asymmetric}
    instance noise : {instance}
    real noise : {clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100 } please refer to http://competition.noisylabels.com/
    '''
    parser.add_argument('--noise_type', type = str, help='clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100, instance, pairflip, symmetric, asymmetric', default='clean')
    parser.add_argument('--noise_rate', type = float, default = 0.2)
    '''ema
    ema_decay set 0.997
    '''
    parser.add_argument('--ema_decay', default=0.997, type=float, metavar='M',help='decay of model ema')
    parser.add_argument('--trials', default=3, type=int, help='Number of experiments')
    '''Consistency loss 0.9
    Combine with Strong Augmentation
    '''
    parser.add_argument('--aug_type', type = str, choices=['randaug','autoaug'], default='autoaug')
    parser.add_argument('--consistency', type = float, default = 0.)
    '''class balance loss 0.1
    '''
    parser.add_argument('--class_balance', type = float, default = 0.)
    '''Mixup alpha=1
    '''
    parser.add_argument('--alpha', default=0, type=float, help='mixup interpolation coefficient (default: 1)')
    '''label_smoothing 0.1
    '''
    parser.add_argument('--label_smoothing', default=0, type=float, help='label smoothing')

    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    log_path = os.path.join(args.log_path,f'{args.dataset}_{args.noise_type}_{args.noise_rate}')
    tb_log_path = os.path.join(args.tb_log_path,f'{args.dataset}_{args.noise_type}_{args.noise_rate}')

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(tb_log_path):
        os.makedirs(tb_log_path)

    # add new
    logger = logging.getLogger(__name__)   
    logging.basicConfig(level=logging.INFO,
                filename=os.path.join(log_path,f'{args.aug_type}_{args.consistency}_CB_{args.class_balance}_Mix_{args.alpha}_LS_{args.label_smoothing}_{args.net}_bsz_{args.batch_size}_lr_{args.lr}_epoch_{args.epoch}.log'),
                datefmt='%Y/%m/%d %H:%M:%S',
                format='%(module)s - %(message)s')
    logger.info(args)

    acc_list = []
    ema_acc_list = []
    for trial in range(args.trials):
        logger.info('\n'+'*'*88+'\n')
        tb_path = os.path.join(tb_log_path, f'{trial}')
        if not os.path.exists(tb_path):
            os.makedirs(tb_path)
        tb_logger = tb.Logger(logdir=tb_path, flush_secs=2)
        # three trial
        main(args, logger, tb_logger, acc_list, ema_acc_list)

    acc_list = np.asarray(acc_list)
    ema_acc_list = np.asarray(ema_acc_list)

    # acc_list
    acc_mean = np.mean(acc_list)
    acc_var = np.var(acc_list)
    acc_std = np.std(acc_list)

    # ema_list
    ema_mean = np.mean(ema_acc_list)
    ema_var = np.var(ema_acc_list)
    ema_std = np.std(ema_acc_list)

    result_path = os.path.join(log_path, f'{args.aug_type}_{args.consistency}_CB_{args.class_balance}_Mix_{args.alpha}_LS_{args.label_smoothing}_{args.net}_bsz_{args.batch_size}_lr_{args.lr}_epoch_{args.epoch}_mean_std.log')
    with open(result_path, 'w') as f:
        f.write(f"acc list is: {acc_list} \n")
        f.write(f"mean ± std is {acc_mean:.2f}±{acc_std:.2f}\n")
        f.write(f"acc mean is {acc_mean:.2f}, acc var is {acc_var:.2f}, acc std is {acc_std:.2f}\n\n")

        f.write(f"ema acc list is: {ema_acc_list} \n")
        f.write(f"mean ± std is {ema_mean:.2f}±{ema_std:.2f}\n")
        f.write(f"ema acc mean is {ema_mean:.2f}, acc var is {ema_var:.2f}, acc std is {ema_std:.2f}\n")
