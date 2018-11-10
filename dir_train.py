import visdom
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
from utils.ood_metrics import tpr95, detection

import seaborn as sns  # import this after torch or it will break everything
import sys
from models.vgg import VGG
from models.densenet import DenseNet3
from models.wideresnet import WideResNet
from models.resnet import *
from utils.utils import CSVLogger, Cutout, obtain_dirichelets, mse_loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vis = visdom.Visdom()
vis.env = 'confidence_estimation'

conf_histogram = None

dataset_options = ['cifar10', 'svhn', 'cifar100']
model_options = ['resnet', 'wideresnet', 'densenet', 'vgg13']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--dataset', default='cifar10', choices=dataset_options)
parser.add_argument('--model', default='vgg13', choices=model_options)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--option', type=str, default="prior")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--id', type=str, default="default")
parser.add_argument('--learning_rate', type=float, default=0.1)
parser.add_argument('--gradient_norm', type=float, default=0.5)
parser.add_argument('--test', type=bool, default=False)
parser.add_argument('--data_augmentation', action='store_true', default=False,
                    help='augment data by flipping and cropping')
parser.add_argument('--KL', type=float, default=0.01)
parser.add_argument('--cutout', type=int, default=16, metavar='S',
                    help='patch size to cut out. 0 indicates no cutout')
parser.add_argument('--baseline', action='store_true', default=False,
                    help='train model without confidence branch')

args = parser.parse_args()
cudnn.benchmark = True  # Should make training should go faster for large models
filename = args.dataset + '_' + args.model + '_' + args.id

if args.dataset == 'svhn' and args.model == 'wideresnet':
    args.model = 'wideresnet16_8'
elif args.dataset == 'cifar10' and args.model == 'resnet':
    args.model = 'resnet18'
elif args.dataset == 'cifar100' and args.model == 'resnet':
    args.model = 'resnet34'

print args

# Image Preprocessing
if args.dataset == 'svhn':
    normalize = transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
                                     std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
elif args.dataset == 'cifar10':
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
elif args.dataset == 'cifar100':
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                    std=[0.2675, 0.2565, 0.2761])
else:
    raise ValueError("No such dataset")

train_transform = transforms.Compose([])
if args.data_augmentation:
    train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
    if args.dataset != 'svhn':
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(normalize)
if args.cutout > 0:
    train_transform.transforms.append(Cutout(args.cutout))

test_transform = transforms.Compose([transforms.ToTensor(), normalize])

if args.dataset == 'cifar10':
    num_classes = 10
    train_dataset = datasets.CIFAR10(root='data/',
                                     train=True,
                                     transform=train_transform,
                                     download=True)

    test_dataset = datasets.CIFAR10(root='data/',
                                    train=False,
                                    transform=test_transform,
                                    download=True)

elif args.dataset == 'cifar100':
    num_classes = 100
    train_dataset = datasets.CIFAR100(root='data/',
                                   train=True,
                                   transform=train_transform,
                                   download=True)    
    test_dataset = datasets.CIFAR100(root='data/',
                                   train=False,
                                   transform=test_transform,
                                   download=True)
elif args.dataset == 'svhn':
    num_classes = 10
    train_dataset = datasets.SVHN(root='data/',
                                  split='train',
                                  transform=train_transform,
                                  download=True)

    test_dataset = datasets.SVHN(root='data/',
                                 split='test',
                                 transform=test_transform,
                                 download=True)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=2)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=2)


def plot_histograms(corr, conf, bins=50, norm_hist=True):
    # Plot histogram of correctly classified and misclassified examples in visdom
    global conf_histogram

    plt.figure(figsize=(6, 4))
    sns.distplot(conf[corr], kde=False, bins=bins, norm_hist=norm_hist, label='Correct')
    sns.distplot(conf[np.invert(corr)], kde=False, bins=bins, norm_hist=norm_hist, label='Incorrect')
    plt.xlabel('Confidence')
    plt.ylabel('Density')
    plt.legend()

    # the image buffer acts as if it where a location on disk
    img_buffer = BytesIO()
    plt.savefig(img_buffer, bbox_inches='tight', pad_inches=0)
    img = Image.open(img_buffer)
    img = img.convert('RGB')
    img = torch.FloatTensor(np.array(img)).permute(2, 0, 1)
    conf_histogram = vis.image(img, win=conf_histogram, opts=dict(title='Confidence Histogram'))

def test(loader):
    cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).

    correct = []
    probability = []
    confidence = []

    for images, labels in loader:
        with torch.no_grad():            
            images = Variable(images).to(device)
        labels = labels.to(device)
        pred, _ = cnn(images)
        alphas, _, conf = obtain_dirichelets(pred, mean=False)
        pred_value, pred = torch.max(pred.data, 1)
        correct.extend((pred == labels).cpu().numpy())
        probability.extend(pred_value.cpu().numpy())
        confidence.extend(conf.cpu().data.numpy())

    correct = np.array(correct).astype(bool)
    probability = np.array(probability)
    confidence = np.array(confidence)

    #plot_histograms(correct, probability)

    val_acc = np.mean(correct)
    conf_min = np.min(confidence)
    conf_max = np.max(confidence)
    conf_avg = np.mean(confidence)

    cnn.train()
    return val_acc, conf_min, conf_max, conf_avg

def ood_evaluate():
    cnn.eval()

    ind_loader = test_loader

    ood_dataset = datasets.ImageFolder(root='data/iSUN', 
                                    transform=test_transform)
    ood_loader = torch.utils.data.DataLoader(dataset=ood_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=2)

    def evaluate(data_loader, in_dis=True):
        out = []
        for data in data_loader:
            if type(data) == list:
                images, labels = data
            else:
                images = data

            images = Variable(images, requires_grad=True).to(device)
            images.retain_grad()

            pre_logits, _ = cnn(images)
            alphas, _, confidence = obtain_dirichelets(pre_logits, mean=False)

            confidence = confidence.data.cpu().numpy()
            out.append(confidence)
        
        out = np.concatenate(out)
        return out

    ood_scores = evaluate(ood_loader, in_dis=False)
    ood_labels = np.zeros(ood_scores.shape[0])

    ind_scores = evaluate(ind_loader, in_dis=False)
    ind_labels = np.ones(ind_scores.shape[0])

    labels = np.concatenate([ind_labels, ood_labels])
    scores = np.concatenate([ind_scores, ood_scores])

    fpr_at_95_tpr = tpr95(ind_scores, ood_scores)

    cnn.train()
    return fpr_at_95_tpr


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
cnn = torch.nn.DataParallel(cnn)
cnn.to(device)

if args.dataset == "cifar10" or args.dataset == "cifar100":
    cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate, nesterov=True, momentum=0.9, weight_decay=5e-4)
else:
    cnn_optimizer = torch.optim.Adam(cnn.parameters(), lr=args.learning_rate, weight_decay=5e-4)

if args.dataset == 'svhn':
    scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120], gamma=0.1)
else:
    if args.model == 'resnet':
        scheduler = MultiStepLR(cnn_optimizer, milestones=[90, 150, 180], gamma=0.2)
    else:
        scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120, 160], gamma=0.2)

if args.model == 'densenet':
    cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True, weight_decay=1e-4)
    scheduler = MultiStepLR(cnn_optimizer, milestones=[150, 225], gamma=0.1)

criterion = torch.nn.CrossEntropyLoss().to(device)
csv_logger = CSVLogger(args=args, filename='logs/' + filename + '.csv',
                       fieldnames=['epoch', 'train_acc', 'test_acc'])

file_path = 'checkpoints/' + filename + '.pt'
if os.path.exists(file_path):
    pretrained_dict = torch.load(file_path)
    cnn.load_state_dict(pretrained_dict)
    print("reloading model from {}".format(file_path))

def isnan(x):
    return x != x

# Start with a reasonable guess for lambda
for epoch in range(1, args.epochs):
    xentropy_loss_avg = 0.
    kl_loss_avg = 0.
    confidence_avg = 0.
    correct_count = 0.
    total = 0.
    confidence_loss_avg = 0.

    progress_bar = tqdm(train_loader)
    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))

        images = Variable(images).to(device)
        labels = Variable(labels).to(device)

        cnn.zero_grad()

        pred_original, _ = cnn(images)

        alphas, _, confidence = obtain_dirichelets(pred_original, mean=True)

        train_loss, train_loss_KL = mse_loss(labels, alphas, args.KL, option=args.option)
        xentropy_loss = torch.mean(train_loss + train_loss_KL)

        total_loss = xentropy_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(cnn.parameters(), args.gradient_norm)

        cnn_optimizer.step()

        xentropy_loss_avg += torch.mean(train_loss).item()
        kl_loss_avg += torch.mean(train_loss_KL).item()
        confidence_avg += confidence.item()

        pred_idx = torch.argmax(pred_original.data, 1)
        total += labels.size(0)

        correct_count += (pred_idx == labels.data).sum()
        accuracy = correct_count.item() / total

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            KL_loss='%.4f' % (kl_loss_avg / (i + 1)),
            confidence_score='%.3f' % (confidence_avg / (i + 1)),
            acc='%.3f' % accuracy)

    test_acc, conf_min, conf_max, conf_avg = test(test_loader)
    tqdm.write('test_acc: %.3f, conf_min: %.3f, conf_max: %.3f, conf_avg: %.3f' % (test_acc, conf_min, conf_max, conf_avg))

    fpr = ood_evaluate()
    tqdm.write("FPR at TPR=95%%: %.3f" % (fpr))

    scheduler.step(epoch)

    row = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_acc)}
    csv_logger.writerow(row)

    torch.save(cnn.state_dict(), 'checkpoints/' + filename + '.pt')

csv_logger.close()
