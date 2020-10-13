import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import torch.nn.functional as F
import numpy as np


def CrossEntropy(outputs, targets, T=3):
    log_softmax_outputs = F.log_softmax(outputs/T, dim=1)
    softmax_targets = F.softmax(targets/T, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()

def get_orth_loss(net):
    orth = 0
    for layer in net.link:
        para = list(layer.parameters())[0]
        reshape_para = para.view(para.shape[0], -1).cuda()
        ATA = torch.mm(reshape_para.t(), reshape_para).cuda()
        O = torch.eye(ATA.shape[0]).cuda()
        orth += ((ATA-O)**2).sum().cuda()
        AAT = torch.mm(reshape_para, reshape_para.t()).cuda()
        O = torch.eye(AAT.shape[0]).cuda()
        orth += ((AAT - O)**2).sum().cuda()
    return orth