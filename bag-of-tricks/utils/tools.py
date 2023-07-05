import argparse
import torch
import numpy as np
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from copy import deepcopy
from torch.autograd import Variable
from collections import OrderedDict
from sklearn.metrics import average_precision_score


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()

        # import ipdb; ipdb.set_trace()

        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

def set_seed(seed):
    # Seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True

def accuracy_topK(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def accuracy(model, loader):
    model.eval()
    total = correct = 0
    for images, labels, _, indexes in loader:
        images = images[0].cuda()
        output = model(images)
        output = F.softmax(output, dim=1)
        pred = torch.argmax(output.data, dim=1)
        total += len(indexes)
        correct += (pred.cpu() == labels).sum().item()
    acc = 100 * float(correct) / float(total)
    return acc

def evaluate(model, loader):
    model.eval()  # Change model to 'eval' mode.
    correct = 0
    total = 0
    for images, _, trues, indexes in loader:
        images = images[0].cuda()
        logits = model(images)
        outputs = F.softmax(logits, dim=1)
        pred = torch.argmax(outputs.data, 1)
        total += trues.size(0)
        correct += (pred.cpu() == trues).sum().item()
    acc = 100 * float(correct) / float(total)
    return acc

class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

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
        self.avg = self.sum / (self.count + 1e-20)
