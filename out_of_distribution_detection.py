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
import os
import seaborn as sns
from models.vgg import VGG, VarVGG
from models.vgg19 import vgg19_bn
from models.densenet import DenseNet3
from models.wideresnet import WideResNet
from models.resnet import *
from utils.ood_metrics import tpr95, detection
from utils.datasets import GaussianNoise, UniformNoise
from utils.utils import obtain_dirichelets, cosine_distance, mse_loss, Transit

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
parser.add_argument('--epsilon', type=float, default=0.005, help='Noise magnitude')
parser.add_argument('--checkpoint', default='cifar10_vgg13_budget_0.3_seed_0', type=str,
                    help='filepath for checkpoint to load')

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
elif args.model == 'vgg19':
    cnn = vgg19_bn(num_classes=num_classes)
elif args.model == 'varvgg13':
    cnn = VarVGG(vgg_name='VGG13', num_classes=num_classes)

cnn = torch.nn.DataParallel(cnn)
cnn.to(device)

proj = Transit(num_classes, num_classes, bias=False)
proj.to(device)

pretrained_dict = torch.load('checkpoints/' + filename + '.pt')
if "state_dict" in pretrained_dict:
    pretrained_dict = pretrained_dict['state_dict']
cnn.load_state_dict(pretrained_dict)

proj_filename = 'checkpoints/' + args.ind_dataset + '_proj.pt'
if os.path.exists(proj_filename):
    pretrained_dict = torch.load(proj_filename)
    proj.load_state_dict(pretrained_dict)

cnn.eval()

#f = lambda x: torch.log(x+1)
f = lambda x: torch.relu(x)
##############################################
### Evaluate out-of-distribution detection ###
##############################################
def evaluate(data_loader, mode):
    out = []
    all_alphas = []
    xent = nn.CrossEntropyLoss()
    for data in data_loader:
        if type(data) == list:
            images, labels = data
        else:
            images = data

        images = Variable(images, requires_grad=True).cuda()
        images.retain_grad()

        if mode == 'dirichlet':
            cnn.zero_grad()
            pre_logits, _ = cnn(images)
            alphas, _, confidence = obtain_dirichelets(pre_logits, func=f, mean=False)
            _, pred_idx = torch.max(pre_logits.data, 1)
            loss = torch.mean(mse_loss(Variable(pred_idx), alphas, 0, option=args.option)[0])
            loss.backward()
            images = images - args.epsilon * torch.sign(images.grad)
            
            pre_logits, _ = cnn(images)
            alphas, evidence, confidence = obtain_dirichelets(pre_logits, func=f, mean=False)
            
            all_alphas.append(evidence.view(-1).data.cpu().numpy())
            confidence = confidence.data.cpu().numpy()
            out.append(confidence)
            
        elif mode == 'finetune':
            pre_logits, _ = cnn(images)
            logits = proj(pre_logits)
            alphas, _, confidence = obtain_dirichelets(logits, func=lambda x:x, mean=False)
            confidence = confidence.data.cpu().numpy()
            out.append(confidence)

        elif mode == 'variational':
            cnn.train()
            images = images.repeat(100, 1, 1, 1)
            pre_logits , _ = cnn(images)
            probs = nn.functional.softmax(pre_logits, -1).view(100, args.batch_size, -1)

            avg_probs = probs.mean(0)
            entropy_avg = -torch.sum(avg_probs * torch.log(avg_probs), -1)

            avg_entropy = -torch.sum(probs * torch.log(probs), -1).mean(0)

            conf = -(entropy_avg - avg_entropy).data.cpu().numpy()
            out.append(conf)
            if len(out) > 100:
                break
            else:
                print("finished {}".format(len(out)))

        elif mode == 'confidence':
            _, confidence = cnn(images)
            confidence = F.sigmoid(confidence)
            confidence = confidence.data.cpu().numpy()
            out.append(confidence)

        elif mode == 'confidence_scaling':
            epsilon = args.epsilon

            cnn.zero_grad()
            _, confidence = cnn(images)
            confidence = F.sigmoid(confidence).view(-1)
            loss = torch.mean(-torch.log(confidence))
            loss.backward()

            images = images - args.epsilon * torch.sign(images.grad)
            images = Variable(images.data, requires_grad=True)

            _, confidence = cnn(images)
            confidence = F.sigmoid(confidence)
            confidence = confidence.data.cpu().numpy()
            out.append(confidence)

        elif mode == 'baseline':
            # https://arxiv.org/abs/1610.02136
            pred, _ = cnn(images)
            pred = F.softmax(pred, dim=-1)
            pred = torch.max(pred.data, 1)[0]
            #pred = torch.sum(pred * torch.log(pred), -1).data
            pred = pred.cpu().numpy()
            out.append(pred)

        elif mode == 'ODIN':
            # https://arxiv.org/abs/1706.02690
            T = args.T
            epsilon = args.epsilon

            cnn.zero_grad()
            pred, _ = cnn(images)
            _, pred_idx = torch.max(pred.data, 1)
            labels = Variable(pred_idx)
            pred = pred / T
            loss = xent(pred, labels)
            loss.backward()
            
            images = images - epsilon * torch.sign(images.grad)
            images = Variable(images.data, requires_grad=True)

            pred, _ = cnn(images)

            pred = pred / T
            pred = F.softmax(pred, dim=-1)
            pred = torch.max(pred.data, 1)[0]
            pred = pred.cpu().numpy()
            out.append(pred)

    out = np.concatenate(out)
    """
    all_alphas = np.concatenate(all_alphas)
    plt.figure()
    sns.distplot(all_alphas, kde=False, bins=100, norm_hist=False)
    plt.title("Calibrated concentration distribution", fontsize=18)
    plt.xlabel("Magnitude of alpha")
    plt.ylabel("Density") 
    plt.savefig("./alpha_distribution_calibrated.eps", format='eps', dpi=1000)
    sys.exit(0)
    """
    return out

ind_scores = evaluate(ind_loader, args.process)
ind_labels = np.ones(ind_scores.shape[0])

ood_scores = evaluate(ood_loader, args.process)
ood_labels = np.zeros(ood_scores.shape[0])

labels = np.concatenate([ind_labels, ood_labels])
scores = np.concatenate([ind_scores, ood_scores])

fpr_at_95_tpr = tpr95(ind_scores, ood_scores)
detection_error, best_delta = detection(ind_scores, ood_scores)
auroc = metrics.roc_auc_score(labels, scores)
aupr_in = metrics.average_precision_score(labels, scores)
aupr_out = metrics.average_precision_score(-1 * labels + 1, 1 - scores)

print("")
print("Method: " + args.process)
print("TPR95 (lower is better): ", fpr_at_95_tpr)
print("Detection error (lower is better): ", detection_error)
print("Best threshold:", best_delta)
print("AUROC (higher is better): ", auroc)
print("AUPR_IN (higher is better): ", aupr_in)
print("AUPR_OUT (higher is better): ", aupr_out)

ranges = (np.min(scores), np.max(scores))
sns.distplot(ind_scores.ravel(), hist_kws={'range': ranges}, kde=False, bins=100, norm_hist=True, label='In-distribution')
sns.distplot(ood_scores.ravel(), hist_kws={'range': ranges}, kde=False, bins=100, norm_hist=True, label='Out-of-distribution')
plt.title("Non-calibrated Dirichlet confidence", fontsize=18)
plt.xlabel('Confidence')
plt.ylabel('Density')
plt.legend()
#plt.title(args.ood_dataset)
#plt.xlim(0, 150)
#plt.savefig("./{}_{}_readjust_noncalibrate.eps".format(args.model, args.ood_dataset), format='eps', dpi=1000)
plt.show()
