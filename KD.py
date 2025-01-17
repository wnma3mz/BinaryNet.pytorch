# coding: utf-8
import argparse
import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import models
from torch.autograd import Variable
from data import get_dataset
from preprocess import get_transform
from utils import *
from datetime import datetime
from ast import literal_eval
from torchvision.utils import save_image
import torch.nn.functional as F


def evlation(net, testloader):
    # 评估
    net.eval()

    # top1 = AverageMeter()
    # top5 = AverageMeter()
    # """
    # time_start = time.time()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.cuda(), labels.cuda()
            # images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)  # 总样本数
            correct += (predicted == labels).sum().item()  # 预测正确数

    acc = 100 * correct / total
    # """
    """
    for i, (inputs, target) in enumerate(data_loader):
        
        input_var = Variable(inputs.type(args.type), volatile=not training)
        target_var = Variable(target)
    
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        if type(output) is list:
            output = output[0]
    
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        # losses.update(loss.item(), inputs.size(0)) 
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))
        # top1.avg
    """

    # time_end = time.time()
    # print('Time cost:', time_end - time_start, "s")
    return acc


if __name__ == '__main__':
    # hyper params
    input_size = None
    # imagenet
    dataset = 'cifar100'
    results_dir = './KD_results'
    save = 'resnet18_binary'

    # alexnet
    MODEL = 'resnet_binary'
    T_MODEL = 'resnet_bin'
    MODEL_CONFIG = ''
    print_freq = 10

    t_net_name = ''

    TYPE = 'torch.cuda.FloatTensor'
    gpus = '2'
    workers = 8
    epochs = 2500
    start_epoch = 0
    batch_size = 256
    optimizer = 'SGD'
    lr = 0.1
    momentum = 0.9
    weight_decay = 1e-4

    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    save_dir = os.path.join(results_dir, save, dataset)

    model = models.__dict__[MODEL]
    """
    # BNN
    t_net = models.__dict__[T_MODEL]
    t_net_config = {'input_size': input_size, 'dataset': dataset, 'depth': 34}
    t_net = t_net(**t_net_config)
    t_net.load_state_dict(torch.load(t_net_name))
    t_net.cuda()
    """

    # Generally
    t_net = resnet50(pretrained=True)
    t_net.cuda()
    # 修改最后一层全连接层,in_features保持不变. num_classes输出需对应分类数目，比如cifar10为10，cifar100为100
    num_fits = t_net.fc.in_features
    t_net.fc = Linear(num_fits, 100)

    model_config = {'input_size': input_size, 'dataset': dataset}

    if MODEL_CONFIG is not '':
        # literal_eval -> eval高级版，进行了安全检查
        model_config = dict(model_config, **literal_eval(MODEL_CONFIG))

    model = model(**model_config)
    model.cuda()
    criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)()
    # criterion.type(TYPE)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    default_transform = {
        'train': get_transform(dataset, input_size=input_size, augment=True),
        'eval': get_transform(dataset, input_size=input_size, augment=False)
    }
    transform = getattr(model, 'input_transform', default_transform)
    regime = getattr(
        model, 'regime', {
            0: {
                'optimizer': optimizer,
                'lr': lr,
                'momentum': momentum,
                'weight_decay': weight_decay
            }
        })

    train_data = get_dataset(dataset, 'train', transform['train'])
    test_data = get_dataset(dataset, 'eval', transform['eval'])

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=workers,
                                               pin_memory=False)

    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=workers,
                                              pin_memory=False)

    # model.train()
    # acc = 0.

    t_net.eval()
    model.train()
    temperature = 1
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    acc = 0.
    for epoch in range(start_epoch, epochs):
        optimizer = adjust_optimizer(optimizer, epoch, regime)

        for i, (inputs, target) in enumerate(train_loader):
            # measure data loading time
            # data_time.update(time.time() - end)
            input_var, _ = inputs.cuda(), target.cuda()

            # compute output
            output = model(input_var)
            out_s = output
            out_t = t_net(input_var)

            loss = -(F.softmax(out_t / temperature, 1).detach() *
                     (F.log_softmax(out_s / temperature, 1) -
                      F.log_softmax(out_t / temperature, 1).detach())).sum()
            if type(output) is list:
                output = output[0]

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            for p in list(model.parameters()):
                if hasattr(p, 'org'):
                    p.data.copy_(p.org)
            optimizer.step()
            for p in list(model.parameters()):
                if hasattr(p, 'org'):
                    p.org.copy_(p.data.clamp_(-1, 1))

        torch.save(model.state_dict(), os.path.join(save_dir, 'new.pth'))

        now_acc = evlation(model, test_loader)
        if now_acc > acc:
            acc = now_acc
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pth'))
        if epoch % print_freq == 0:
            print('epoch: %d loss: %.4f time: %.4f' %
                  (epoch + 1, loss.item(), time.time() - s1))
            print('Acc: {}'.format(acc))
