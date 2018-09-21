# Dirichlet-Out-of-Distributoin-Detection
This is the code for paper "A smoothed variational dirichlet framework for out-of-distribution detection"

## To run the VGG13 experiment, please use:

```
python dir_train.py --dataset cifar10 --model vgg13 --data_augmentation --id variational_dirichlet --KL 0.01 --gradient_norm 5
```

## To run ResNet experiment, please use:

```
python dir_train.py --dataset cifar10 --model resnet --data_augmentation --id variational_dirichlet --batch_size 1024 --KL 0.01 --gradient_norm 1
```

## To run out-of-distribution detection, please use:

```
CUDA_VISIBLE_DEVICES=1 python out_of_distribution_detection.py --ind_dataset cifar10 --ood_dataset iSUN  --model vgg13 --process dirichlet --checkpoint cifar10_vgg13_baseline_parallel --epsilon 0.005
```
