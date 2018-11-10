import torch
import csv
import numpy as np
import math
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn as nn

class Transit(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Transit, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        #stdv = 1. / math.sqrt(self.weight.size(1))
        #self.weight.data.uniform_(-stdv, stdv)
        self.weight.data.copy_(torch.zeros(self.in_features, self.in_features))
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        weight = torch.clamp(self.weight, min=-1, max=1)
        return F.linear(input, self.weight, self.bias) + input

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

def cosine_distance(x1, x2):
    norm_0 = torch.sqrt(torch.pow(x1, 2).sum(1, True))
    norm_1 = torch.sqrt(torch.pow(x2, 2).sum(0, True))
    norm = torch.mm(norm_0, norm_1)
    dot_product = torch.mm(x1, x2)
    cosine_distance = 0.5 * (1 - dot_product / norm)
    return cosine_distance

def one_hot(seq_batch, depth):
    # seq_batch.size() should be [seq,batch] or [batch,]
    # return size() would be [seq,batch,depth] or [batch,depth]
    out = torch.zeros(seq_batch.size()+torch.Size([depth])).cuda()
    dim = len(seq_batch.size())
    index = seq_batch.view(seq_batch.size()+torch.Size([1]))
    return out.scatter_(dim,index,1)

def mse_loss(label, alpha, max_weight, option="variational"):
    if option == "variational":
        loss = -(torch.digamma(alpha) - torch.digamma(alpha.sum(-1)).unsqueeze(-1))
    elif option == "prior":
        loss = -torch.log(alpha / alpha.sum(-1, keepdim=True))
    selected_loss = loss.gather(1, label.unsqueeze(-1)).squeeze()
   
    mask = 1 - one_hot(label, alpha.size(1))
    appended = one_hot(label, alpha.size(1))
    alpha = alpha * mask + appended

    def KL(alp):
        beta = torch.FloatTensor(1, alpha.size(1)).fill_(1.0).cuda()
        S_alpha = torch.sum(alpha, -1, True)
        S_beta = torch.sum(beta, -1, True)
        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), -1, True)
        lnB_uni = torch.sum(torch.lgamma(beta), -1, True) - torch.lgamma(S_beta)
        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)
        kl = torch.sum((alpha - beta) * (dg1 - dg0), -1, True) + lnB + lnB_uni
        return kl

    return selected_loss, max_weight * KL(alpha)

def entropy(alpha):
    alpha0 = torch.sum(alpha, 1)
    logB = torch.sum(torch.lgamma(alpha), 1) - torch.lgamma(alpha0)
    digamma_1 = (alpha0 - alpha.size(1)) * torch.digamma(alpha0)
    digamma_2 = -torch.sum((alpha - 1) * torch.digamma(alpha), 1)
    entropy = logB + digamma_1 + digamma_2
    return entropy, logB, digamma_1, digamma_2

def obtain_dirichelets(logits, func=lambda x:torch.relu(x), mean=False):
    alpha = torch.exp(logits).clamp(0, 1000)
    evidence = func(alpha) 
    ent, ret1, ret2, ret3 = entropy(evidence)    
    conf = -ent
    if mean:
        conf = torch.mean(conf)
    return alpha, evidence, conf

class CSVLogger():
    def __init__(self, args, filename='log.csv', fieldnames=['epoch']):

        self.filename = filename
        self.csv_file = open(filename, 'w')

        # Write model configuration at top of csv
        writer = csv.writer(self.csv_file)
        for arg in vars(args):
            writer.writerow([arg, getattr(args, arg)])
        writer.writerow([''])

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()


class Cutout(object):
    """Randomly mask out one or more patches from an image.
       https://arxiv.org/abs/1708.04552
    Args:
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        if np.random.choice([0, 1]):
            mask = np.ones((h, w), np.float32)

            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length / 2, 0, h)
            y2 = np.clip(y + self.length / 2, 0, h)
            x1 = np.clip(x - self.length / 2, 0, w)
            x2 = np.clip(x + self.length / 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img = img * mask

        return img
