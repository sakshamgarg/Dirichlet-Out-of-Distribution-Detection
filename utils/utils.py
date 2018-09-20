import torch
import csv
import numpy as np

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

def mse_loss(label, alpha, annealing_step):    
    loss = -(torch.digamma(alpha) - torch.digamma(alpha.sum(-1)).unsqueeze(-1))
    selected_loss = loss.gather(1, label.unsqueeze(-1)).squeeze()
    uncertain_bias = 1.0
    annealing_coef = min(annealing_step / 100.0, 0.01)

    mask = 1 - one_hot(label, alpha.size(1))
    appended = one_hot(label, alpha.size(1))
    alpha = alpha * mask + appended

    def KL(alp):
        beta = torch.FloatTensor(1, alpha.size(1)).fill_(uncertain_bias).cuda()
        S_alpha = torch.sum(alpha, -1, True)
        S_beta = torch.sum(beta, -1, True)
        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), -1, True)
        lnB_uni = torch.sum(torch.lgamma(beta), -1, True) - torch.lgamma(S_beta)
        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)
        kl = torch.sum((alpha - beta) * (dg1 - dg0), -1, True) + lnB + lnB_uni
        return kl

    return selected_loss, annealing_coef * KL(alpha)

def entropy(alpha):
    alpha0 = torch.sum(alpha, 1)
    logB = torch.sum(torch.lgamma(alpha), 1) - torch.lgamma(alpha0)
    digamma_1 = (alpha0 - alpha.size(1)) * torch.digamma(alpha0)
    digamma_2 = -torch.sum((alpha - 1) * torch.digamma(alpha), 1)
    entropy = logB + digamma_1 + digamma_2
    return entropy, logB, digamma_1, digamma_2

def obtain_dirichelets(logits, mean=False):
    alpha = torch.exp(logits)
    evidence = torch.relu(logits) + 1
    ent, ret1, ret2, ret3 = entropy(evidence)
    conf = -ent

    if mean:
        conf = torch.mean(conf)
    return alpha, conf

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
