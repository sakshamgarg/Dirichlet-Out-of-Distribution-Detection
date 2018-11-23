import pdb
import argparse
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
import torch
import os
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torchvision.utils import make_grid
from torch.autograd import Variable
from models.vgg import VGG, VarVGG
from models.resnext import resnext
from models.densenet import DenseNet3
from models.wideresnet import WideResNet
from models.resnet import *
from utils.ood_metrics import tpr95, detection
from utils.datasets import GaussianNoise, UniformNoise
from utils.utils import obtain_dirichelets, cosine_distance, mse_loss, Transit, reproject_dirichlets
from mpl_toolkits.axes_grid1 import make_axes_locatable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ind_options = ['cifar10', 'svhn', 'cifar100']
ood_options = ['tinyImageNet_resize', 'LSUN_resize', 'iSUN', 'cifar10', 'svhn']
model_options = ['densenet', 'wideresnet', 'vgg13', 'vgg19', 'varvgg13', 'resnet', 'resnext']
process_options = ['baseline', 'ODIN', 'confidence', 'confidence_scaling', 'dirichlet', 'variational', \
                   'finetune', 'plot_alpha', 'plot_confidence', 'cross_entropy', 'perturbation_magnitude']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--ind_dataset', default='cifar10', choices=ind_options)
parser.add_argument('--ood_dataset', default='tinyImageNet_resize', choices=ood_options)
parser.add_argument('--model', default='vgg13', choices=model_options)
parser.add_argument('--process', default='confidence', choices=process_options)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--plot', default=False, action='store_true', help='whether to plot the image')
parser.add_argument('--logscale', default=False, action='store_true', help='whether to plot the image')
parser.add_argument('--T', type=float, default=1000., help='Scaling temperature')
parser.add_argument('--epsilon', type=float, default=0.005, help='Noise magnitude')
parser.add_argument('--checkpoint', default='', type=str, help='filepath for checkpoint to load')

args = parser.parse_args()
cudnn.benchmark = True  # Should make training should go faster for large models

filename = args.checkpoint
proj_filename = 'checkpoints/{}_{}_proj.pt'.format(args.ind_dataset, args.model)

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
elif args.ind_dataset in ['cifar10', 'cifar100']:
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
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
elif args.ood_dataset == 'cifar10':
    ood_dataset = datasets.CIFAR10(root='data/', train=False, transform=transform, download=True)
elif args.ood_dataset == 'svhn':
    ood_dataset = datasets.SVHN(root='data/', split='test', transform=transform, download=True)

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

#if hasattr(ood_dataset, 'imgs'):
#ood_loader.dataset.imgs = ood_loader.dataset.imgs[1000:]
#ood_loader.dataset.__len__ = len(ood_loader.dataset.imgs)
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
elif args.model == 'resnext':
    cnn = resnext(cardinality=8, num_classes=num_classes, depth=29, widen_factor=4)
elif args.model == 'vgg13':
    cnn = VGG(vgg_name='VGG13', num_classes=num_classes)
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

if os.path.exists(proj_filename):
    pretrained_dict = torch.load(proj_filename)
    proj.load_state_dict(pretrained_dict)

cnn.eval()

##############################################
### Evaluate out-of-distribution detection ###
##############################################
def evaluate(data_loader, mode):
    out = []
    all_alphas = []
    all_ratios = []
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
            f = lambda x: torch.log(x+1)
            
            alphas, _, confidence = obtain_dirichelets(pre_logits, func=f, mean=False)
            _, pred_idx = torch.max(pre_logits.data, 1)
            loss = torch.mean(mse_loss(Variable(pred_idx), alphas, 0, option="variational")[0])
            loss.backward()
            images = images - args.epsilon * torch.sign(images.grad)
            
            pre_logits, _ = cnn(images)
            alphas, evidence, confidence = obtain_dirichelets(pre_logits, func=f, mean=False)
            
            all_alphas.append(evidence.view(-1).data.cpu().numpy())
            confidence = confidence.data.cpu().numpy()
            out.append(confidence)

        elif mode == 'cross_entropy':
            cnn.zero_grad()
            pre_logits, _ = cnn(images)
            f = lambda x : torch.relu(x)

            _, pred_idx = torch.max(pre_logits.data, 1)
            #loss = xent(pre_logits, pred_idx)
            alphas, evidence, confidence = obtain_dirichelets(pre_logits, func=f, mean=True)
            loss = -confidence
            loss.backward()
            images = images - args.epsilon * torch.sign(images.grad)
            
            pre_logits, _ = cnn(images)
            alphas, evidence, confidence = obtain_dirichelets(pre_logits, func=f, mean=False)
            
            all_alphas.append(evidence.view(-1).data.cpu().numpy())
            confidence = confidence.data.cpu().numpy()
            out.append(confidence)

        elif mode == 'finetune':
            pre_logits, _ = cnn(images)
            confidence, alphas = reproject_dirichlets(pre_logits, proj, args.model, mean=False, logscale=args.logscale)
            #confidence, alphas = reproject_dirichlets(pre_logits, lambda x:x, args.model, mean=False, logscale=args.logscale)

            confidence = confidence.data.cpu().numpy()
            out.append(confidence)

        elif mode == 'perturbation_magnitude':
            pre_logits, _ = cnn(images)
            confidence, alphas = reproject_dirichlets(pre_logits, proj, args.model, mean=False)
            orig_confidence, orig_alphas = reproject_dirichlets(pre_logits, lambda x:x, args.model, mean=False)

            perturb = alphas - orig_alphas
            perturb_norm = torch.max(perturb, -1)[0]#torch.norm(perturb, 2, -1)
            old_norm = torch.max(orig_alphas, -1)[0]#torch.norm(orig_alphas, 2, -1)
            ratio = (perturb_norm / old_norm)
            ratio = ratio.cpu().cpu().data.numpy()
            ratio = ratio[ratio < 0.12]
            all_ratios.append(ratio)

        elif mode == 'plot_alpha':
            pre_logits, _ = cnn(images)
            confidence, alphas = reproject_dirichlets(pre_logits, proj, args.model, mean=False)
            orig_confidence, orig_alphas = reproject_dirichlets(pre_logits, lambda x:x, args.model, mean=False)

            alphas = alphas.cpu().data.numpy()
            orig_alphas = orig_alphas.cpu().data.numpy()
           
            fig, axes = plt.subplots(nrows=1, ncols=2)
            extent = [1, 100, 1, args.batch_size]
            for i, ax in enumerate(axes.flat):
                if i == 0:
                    im = ax.imshow(orig_alphas, extent=extent, interpolation='none')
                    ax.title.set_text("before")
                else:
                    im = ax.imshow(alphas, extent=extent, interpolation='none')
                    ax.title.set_text("after")
                ax.set_xlabel("CIFA100 classes")
                if i == 0:
                    ax.set_ylabel("Samples")

            fig.subplots_adjust(right=0.8)
            cax = fig.add_axes([0.85,0.2,0.03,0.6])
            fig.colorbar(im, cax=cax)
            plt.savefig("images/alpha_projection.eps", format='eps', dpi=200)

            sys.exit(0)

        elif mode == 'plot_confidence':
            pre_logits, _ = cnn(images)
            orig_confidence, orig_alphas = reproject_dirichlets(pre_logits, lambda x:x, args.model, mean=False)

            f = lambda x : torch.relu(x)
            alphas, evidence, confidence = obtain_dirichelets(pre_logits, func=f, mean=True)
            loss = -confidence
            loss.backward()
            images = images - args.epsilon * torch.sign(images.grad)
            pre_logits, _ = cnn(images)
            confidence, alphas = reproject_dirichlets(pre_logits, lambda x:x, args.model, mean=False)

            confidence = confidence.cpu().data.numpy()
            orig_confidence = orig_confidence.cpu().data.numpy()

            plt.hist(orig_confidence, bins=50, label='before perturbation')
            plt.hist(confidence, bins=50, label='after perturbation')
            plt.title("Confidence Shift after Perturbation")
            plt.legend()
            plt.xlabel('Confidence')
            plt.ylabel('Density')

            #fig, axes = plt.subplots(nrows=1, ncols=2)
            #extent = [0.05, 1.0, 1.0, 0.05]
            #for i, ax in enumerate(axes.flat):
            #    if i == 0:
            #        im = ax.hist(orig_confidence, bins=50, density=True)
            #        ax.title.set_text("before")
            #    else:
            #        im = ax.hist(confidence, bins=50, density=True)
            #        ax.title.set_text("after")
            #    ax.set_xlabel("Confidence")
            #plt.tight_layout()

            plt.savefig("images/ood_confidence_projection.eps", format='eps', dpi=200)
            sys.exit(0)

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

    #out = np.concatenate(out)

    all_ratios = np.concatenate(all_ratios)
    plt.figure()
    sns.distplot(all_ratios, kde=False, bins=100, norm_hist=False)
    plt.xlabel("Perturbation Ratio")
    plt.ylabel("Density") 
    plt.savefig("./perturbation_ratio_distribution.eps", format='eps', dpi=1000)
    sys.exit(0)
    return out

ood_scores = evaluate(ood_loader, args.process)
ood_labels = np.zeros(ood_scores.shape[0])
ind_scores = evaluate(ind_loader, args.process)
ind_labels = np.ones(ind_scores.shape[0])

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

if args.plot:
    ranges = (np.min(scores), np.max(scores))
    sns.distplot(ind_scores.ravel(), hist_kws={'range': ranges}, kde=False, bins=100, norm_hist=True, label='In-distribution')
    sns.distplot(ood_scores.ravel(), hist_kws={'range': ranges}, kde=False, bins=100, norm_hist=True, label='Out-of-distribution')
    plt.title("{}/{}".format(args.ind_dataset, args.ood_dataset), fontsize=18)
    plt.xlabel('Confidence')
    plt.ylabel('Density')
    plt.legend()
    #plt.title(args.ood_dataset)
    #plt.xlim(0, 150)
    plt.savefig("images/{}_{}_{}.eps".format(args.model, args.ind_dataset, args.ood_dataset), format='eps', dpi=1000)
    #plt.show()
