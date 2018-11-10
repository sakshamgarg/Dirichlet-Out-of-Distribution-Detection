import pdb
import argparse
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torchvision.utils import make_grid
from torch.autograd import Variable
from itertools import cycle
import seaborn as sns
from models.vgg import VGG, VarVGG
from models.vgg19 import vgg19_bn
from models.densenet import DenseNet3
from models.wideresnet import WideResNet
from models.resnet import *
from utils.ood_metrics import tpr95, detection
from utils.datasets import GaussianNoise, UniformNoise
from utils.utils import obtain_dirichelets, cosine_distance, mse_loss, Transit, entropy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


ind_options = ['cifar10', 'svhn', 'cifar100']
ood_options = ['tinyImageNet_resize', 'LSUN_resize', 'iSUN']
model_options = ['densenet', 'wideresnet', 'vgg13', 'vgg19', 'varvgg13', 'resnet']
process_options = ['baseline', 'ODIN', 'confidence', 'confidence_scaling', 'dirichlet', 'variational', 'finetune']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--ind_dataset', default='cifar10', choices=ind_options)
parser.add_argument('--ood_dataset', default='tinyImageNet_resize', choices=ood_options)
parser.add_argument('--model', default='vgg13', choices=model_options)
parser.add_argument('--process', default='confidence', choices=process_options)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--option', type=str, default="prior")
parser.add_argument('--T', type=float, default=1000., help='Scaling temperature')
parser.add_argument('--dirichlet', default=False, action='store_true', help='Perform Dirichlet Experiments')
parser.add_argument('--epsilon', type=float, default=0.1, help='Noise magnitude')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Noise magnitude')
parser.add_argument('--checkpoint', default='cifar10_vgg13_budget_0.3_seed_0', type=str,
                    help='filepath for checkpoint to load')
parser.add_argument('--validation', action='store_true', default=False,
                    help='only use first 1000 samples from OOD dataset for validation')

#f = lambda x:torch.log(x+1) + 1
f = lambda x:torch.relu(x)
args = parser.parse_args()
cudnn.benchmark = True  # Should make training should go faster for large models

filename = args.checkpoint

if args.ind_dataset == 'svhn' and args.model == 'wideresnet':
    args.model = 'wideresnet16_8'
elif args.ind_dataset == 'cifar10' and args.model == 'resnet':
    args.model = 'resnet18'
elif args.ind_dataset == 'cifar100' and args.model == 'resnet':
    args.model = 'resnet34'

print args

###########################
### Set up data loaders ###
###########################

if args.ind_dataset == 'svhn':
    normalize = transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
                                     std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
elif args.ind_dataset == 'cifar10':
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
elif args.ind_dataset == 'cifar100':
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                    std=[0.2675, 0.2565, 0.2761])
else:
    raise ValueError("No such dataset")

transform = transforms.Compose([transforms.ToTensor(),
                                normalize])

if args.ind_dataset == 'cifar10':
    num_classes = 10
    ind_dataset = datasets.CIFAR10(root='data/',
                                   train=False,
                                   transform=transform,
                                   download=True)
elif args.ind_dataset == 'cifar100':
    num_classes = 100
    ind_dataset = datasets.CIFAR100(root='data/',
                                   train=False,
                                   transform=transform,
                                   download=True)
elif args.ind_dataset == 'svhn':
    num_classes = 10
    ind_dataset = datasets.SVHN(root='data/',
                                split='test',
                                transform=transform,
                                download=True)

data_path = 'data/'

if args.ood_dataset == 'tinyImageNet_resize':
    ood_dataset = datasets.ImageFolder(root=data_path + 'TinyImagenet_resize', transform=transform)
elif args.ood_dataset == 'LSUN_resize':
    ood_dataset = datasets.ImageFolder(root=data_path + 'LSUN_resize', transform=transform)
elif args.ood_dataset == 'iSUN':
    ood_dataset = datasets.ImageFolder(root=data_path + 'iSUN', transform=transform)

ind_loader = torch.utils.data.DataLoader(dataset=ind_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         pin_memory=True,
                                         num_workers=2)

ood_loader = torch.utils.data.DataLoader(dataset=ood_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         pin_memory=True,
                                         num_workers=2)
valid_ood_loader = torch.utils.data.DataLoader(dataset=ood_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         pin_memory=True,
                                         num_workers=2)

# Limit dataset to first 1000 samples for validation and fine-tuning
# Based on validation procedure from https://arxiv.org/abs/1706.02690
valid_ood_loader.dataset.imgs = ood_loader.dataset.imgs[:1000]
valid_ood_loader.dataset.__len__ = 1000

ood_loader.dataset.imgs = ood_loader.dataset.imgs[1000:]
ood_loader.dataset.__len__ = len(ood_loader.dataset.imgs)

##############################
### Load pre-trained model ###
##############################

if args.model == 'resnet18':
    cnn = ResNet18(num_classes=num_classes)
elif args.model == 'resnet34':
    cnn = ResNet34(num_classes=num_classes)
elif args.model == 'wideresnet':
    cnn = WideResNet(depth=28, num_classes=num_classes, widen_factor=10)
elif args.model == 'wideresnet16_8':
    cnn = WideResNet(depth=16, num_classes=num_classes, widen_factor=8)
elif args.model == 'densenet':
    cnn = DenseNet3(depth=100, num_classes=num_classes, growth_rate=12, reduction=0.5)
elif args.model == 'vgg13':
    cnn = VGG(vgg_name='VGG13', num_classes=num_classes)
elif args.model == 'varvgg13':
    cnn = VarVGG(vgg_name='VGG13', num_classes=num_classes)
elif args.model == 'vgg19':
    cnn = vgg19_bn(num_classes=num_classes)

cnn = torch.nn.DataParallel(cnn)
cnn.to(device)

pretrained_dict = torch.load('checkpoints/' + filename + '.pt')
if "state_dict" in pretrained_dict:
    pretrained_dict = pretrained_dict['state_dict']
cnn.load_state_dict(pretrained_dict)

cnn.eval()

#proj = nn.Sequential(nn.Linear(num_classes, num_classes, bias=True),
#                     nn.ReLU(inplace=True),
#                     nn.Linear(num_classes, num_classes, bias=True),
#                     nn.ReLU(inplace=True),
#                     nn.Linear(num_classes, num_classes, bias=True))
proj = Transit(num_classes, num_classes, bias=False)
#proj.weight.data.copy_(torch.eye(num_classes, num_classes))
#proj.bias.data.copy_(torch.zeros(num_classes, ))
proj.to(device)

optimizer = torch.optim.Adam(proj.parameters(), lr=args.learning_rate, weight_decay=1e-4)
#optimizer = torch.optim.Adam(cnn.parameters(), lr=args.learning_rate, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
criterion.to(device)

def fgsm(cnn, images, labels, targeted=False, eps=args.epsilon, x_val_min=-1, x_val_max=1):
    images_adv = Variable(images.data, requires_grad=True).cuda()
    images_adv.retain_grad()
    pre_logits, _ = cnn(images_adv)
    cost = -criterion(pre_logits, labels)
    
    cnn.zero_grad()
    if images_adv.grad is not None:
        images_adv.grad.data.fill_(0)
    cost.backward()

    images_adv.grad.sign_()
    images_adv = images_adv - eps * images_adv.grad
    images_adv = torch.clamp(images_adv, x_val_min, x_val_max)
    return images_adv

def output_conf(images, mean=False):
    pre_logits, _ = cnn(images)
    inp = Variable(pre_logits.data)
    pre_logits = torch.relu(proj(inp))
    #pre_logits = proj(inp)
    alphas, _, confidence = obtain_dirichelets(pre_logits, func=f, mean=mean)
    return confidence, pre_logits

def calc_fpr():
    def evaluate(data_loader, mode):
        out = []
        all_alphas = []
        for data in data_loader:
            if type(data) == list:
                images, labels = data
            else:
                images = data

            images = Variable(images, requires_grad=True).cuda()
            images.retain_grad()

            if mode == 'dirichlet':
                cnn.zero_grad()
                
                confidence, _ = output_conf(images, mean=False)
                confidence = confidence.data.cpu().numpy()
                out.append(confidence)

        out = np.concatenate(out)
        return out

    ind_scores = evaluate(ind_loader, args.process)
    ind_labels = np.ones(ind_scores.shape[0])

    ood_scores = evaluate(ood_loader, args.process)
    ood_labels = np.zeros(ood_scores.shape[0])

    labels = np.concatenate([ind_labels, ood_labels])
    scores = np.concatenate([ind_scores, ood_scores])

    fpr_at_95_tpr = tpr95(ind_scores, ood_scores)
    print("TPR95 (lower is better): ", fpr_at_95_tpr)

train_ind_loader = cycle(iter(ind_loader))
#train_ood_loader = cycle(iter(valid_ood_loader))
train_ood_loader = None

for i in range(2000):
    def obtain_conf(data, adv_data=None):
        if type(data) == list:
            images, labels = data
        else:
            images = data

        labels = Variable(labels).cuda()

        if adv_data is None:
            adv_images = fgsm(cnn, images, labels)
        else:
            adv_images, _ = adv_data
        
        confidence, pre_logits = output_conf(images, mean=True)
        adv_confidence, _ = output_conf(adv_images, mean=True)

        cross_entropy = criterion(pre_logits, labels)

        return confidence, adv_confidence, cross_entropy

    ind_data = next(train_ind_loader)
    #ood_data = next(train_ood_loader)
    conf, adv_conf, cross_entropy = obtain_conf(ind_data, None)
    loss = adv_conf - conf
    #loss = cross_entropy
    loss.backward()
    optimizer.step()
    #print("loss = {}\r".format(loss))
    if i % 20 == 0:
        calc_fpr()
        proj_filename = 'checkpoints/' + args.ind_dataset + '_proj.pt'
        torch.save(proj.state_dict(), proj_filename)
