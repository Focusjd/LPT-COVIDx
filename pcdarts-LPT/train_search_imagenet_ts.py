import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import copy

from torch.autograd import Variable
from model_search_imagenet import Network
from architect import Architect
from teacher import *


parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--workers', type=int, default=4,
                    help='number of workers to load dataset')
parser.add_argument('--data', type=str, default='/tmp/cache/',
                    help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=768, help='batch size')
parser.add_argument('--learning_rate', type=float,
                    default=0.5, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float,
                    default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float,
                    default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float,
                    default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=50,
                    help='num of training epochs')
parser.add_argument('--init_channels', type=int,
                    default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8,
                    help='total number of layers')
parser.add_argument('--model_path', type=str,
                    default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true',
                    default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int,
                    default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float,
                    default=0.3, help='drop path probability')
parser.add_argument('--save', type=str,
                    default='/tmp/checkpoints/', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float,
                    default=5, help='gradient clipping')
parser.add_argument('--unrolled', action='store_true',
                    default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float,
                    default=6e-3, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float,
                    default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--begin', type=int, default=35, help='batch size')

parser.add_argument('--tmp_data_dir', type=str,
                    default='../../data/', help='temp data dir')
parser.add_argument('--note', type=str, default='try',
                    help='note for this run')

# new hyperparams.
parser.add_argument('--weight_gamma', type=float, default=1.0)
parser.add_argument('--weight_lambda', type=float, default=1.0)
parser.add_argument('--model_v_learning_rate', type=float, default=6e-3)
parser.add_argument('--model_v_weight_decay', type=float, default=1e-3)
parser.add_argument('--learning_rate_w', type=float, default=0.5)
parser.add_argument('--learning_rate_h', type=float, default=0.5)
parser.add_argument('--weight_decay_w', type=float, default=3e-4)
parser.add_argument('--weight_decay_h', type=float, default=3e-4)
parser.add_argument('--is_parallel', type=int, default=0)
parser.add_argument('--teacher_arch', type=str, default='18')
# parser.add_argument('--is_cifar100', type=int, default=0)
args = parser.parse_args()

args.save = '{}search-{}-{}'.format(args.save,
                                    args.note, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

data_dir = os.path.join(args.tmp_data_dir, 'imagenet_sampled')
# data preparation, we random sample 10% and 2.5% from training set(each class) as train and val, respectively.
# Note that the data sampling can not use torch.utils.data.sampler.SubsetRandomSampler as imagenet is too large
CLASSES = 1000


def main():
    if args.teacher_arch == '18':
        teacher_w = resnet18().cuda()
    elif args.teacher_arch == '34':
        teacher_w = resnet34().cuda()
    elif args.teacher_arch == '50':
        teacher_w = resnet50().cuda()
    elif args.teacher_arch == '101':
        teacher_w = resnet101().cuda()
    teacher_h = nn.Linear(512 * teacher_w.block.expansion,
                          CLASSES).cuda()
    teacher_v = nn.Linear(512 * teacher_w.block.expansion, 2).cuda()
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    # torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    #logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    #dataset_dir = '/cache/'
    # pre.split_dataset(dataset_dir)
    # sys.exit(1)
   # dataset prepare
    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    # dataset split
    train_data1 = dset.ImageFolder(traindir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
    train_data2 = dset.ImageFolder(valdir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
    valid_data = dset.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
    num_train = len(train_data1)
    num_val = len(train_data2)
    print('# images to train network: %d' % num_train)
    print('# images to validate network: %d' % num_val)

    model = Network(args.init_channels, CLASSES, args.layers, criterion)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    teacher_w = torch.nn.DataParallel(teacher_w)
    teacher_v = torch.nn.DataParallel(teacher_v)
    teacher_h = torch.nn.DataParallel(teacher_h)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    optimizer_a = torch.optim.Adam(model.module.arch_parameters(),
                                   lr=args.arch_learning_rate, betas=(
                                       0.5, 0.999),
                                   weight_decay=args.arch_weight_decay)
    optimizer_w = torch.optim.SGD(
        teacher_w.parameters(),
        args.learning_rate_w,
        momentum=args.momentum,
        weight_decay=args.weight_decay_w)
    optimizer_h = torch.optim.SGD(
        teacher_h.parameters(),
        args.learning_rate_h,
        momentum=args.momentum,
        weight_decay=args.weight_decay_h)
    optimizer_v = torch.optim.Adam(
        teacher_v.parameters(),
        lr=args.model_v_learning_rate,
        betas=(0.5, 0.999),
        weight_decay=args.model_v_weight_decay)

    # test_queue = torch.utils.data.DataLoader(
    #     valid_data,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     pin_memory=True,
    #     num_workers=args.workers)

    train_queue = torch.utils.data.DataLoader(
        train_data1, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, num_workers=args.workers)

    valid_queue = torch.utils.data.DataLoader(
        train_data2, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, num_workers=args.workers)

    external_queue = torch.utils.data.DataLoader(
        train_data2, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, num_workers=args.workers)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    scheduler_w = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_w, float(args.epochs), eta_min=args.learning_rate_min)
    scheduler_h = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_h, float(args.epochs), eta_min=args.learning_rate_min)

    #architect = Architect(model, args)
    lr = args.learning_rate
    lr_w = args.learning_rate_w
    lr_h = args.learning_rate_h
    for epoch in range(args.epochs):

        current_lr = scheduler.get_lr()[0]
        current_lr_w = scheduler_w.get_lr()[0]
        current_lr_h = scheduler_h.get_lr()[0]
        logging.info('Epoch: %d lr: %e lr_w: %e lr_h: %e', epoch, current_lr,
                     current_lr_w, current_lr_h)
        if epoch < 5 and args.batch_size > 256:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (epoch + 1) / 5.0
            logging.info('Warming-up Epoch: %d, LR: %e',
                         epoch, lr * (epoch + 1) / 5.0)
            print(optimizer)
        genotype = model.module.genotype()
        logging.info('genotype = %s', genotype)
        arch_param = model.module.arch_parameters()
        logging.info(F.softmax(arch_param[0], dim=-1))
        logging.info(F.softmax(arch_param[1], dim=-1))
        # training
        train_acc, train_obj = train(
            train_queue,
            valid_queue,
            external_queue,
            model,
            teacher_w,
            teacher_h,
            teacher_v,
            optimizer,
            optimizer_a,
            optimizer_w,
            optimizer_h,
            optimizer_v,
            criterion,
            lr,
            lr_w,
            lr_h,
            epoch)
        logging.info('Train_acc %f', train_acc)
        scheduler.step()
        scheduler_w.step()
        scheduler_h.step()
        # validation
        if epoch >= 47:
            valid_acc, valid_obj = infer(valid_queue, model, criterion)
            logging.info('Valid_acc %f', valid_acc)


def train(train_queue,
          valid_queue,
          external_queue,
          model,
          model_w,
          model_h,
          model_v,
          optimizer,
          optimizer_a,
          optimizer_w,
          optimizer_h,
          optimizer_v,
          criterion,
          lr,
          lr_w,
          lr_h,
          epoch):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # get a random minibatch from the search queue with replacement
        try:
            input_search, target_search = next(valid_queue_iter)
        except:
            valid_queue_iter = iter(valid_queue)
            input_search, target_search = next(valid_queue_iter)
        input_search = input_search.cuda(non_blocking=True)
        target_search = target_search.cuda(non_blocking=True)
        # for the external data.
        try:
            input_external, target_external = next(external_queue_iter)
        except:
            external_queue_iter = iter(external_queue)
            input_external, target_external = next(external_queue_iter)
        input_external = input_external.cuda(non_blocking=True)
        target_external = target_external.cuda(non_blocking=True)

        if epoch >= args.begin:
            # optimizer_a.zero_grad()
            # logits = model(input_search)
            # loss_a = criterion(logits, target_search)
            # loss_a.sum().backward()
            # nn.utils.clip_grad_norm_(
            #     model.module.arch_parameters(), args.grad_clip)
            # optimizer_a.step()
            optimizer_a.zero_grad()
            # logits = model(input_search)
            # loss_a = criterion(logits, target_search)
            # loss_a.backward()
            logits_external = model(input_external)
            loss_a = F.cross_entropy(
                logits_external, target_external, reduction='none')
            binary_scores_external = model_v(model_w(input_external))
            binary_weight_external = F.softmax(binary_scores_external, 1)
            loss_a = binary_weight_external[:, 1] * loss_a
            loss_a = loss_a.mean()
            loss_a.backward()
            nn.utils.clip_grad_norm_(
                model.module.arch_parameters(), args.grad_clip)
            optimizer_a.step()

            optimizer_v.zero_grad()
            teacher_logits = model_h(model_w(input_search))
            left_loss = args.weight_lambda * criterion(
                teacher_logits, target_search)

            model_logits_external = model(input_external)
            right_loss = F.cross_entropy(
                model_logits_external, target_external, reduction='none')
            binary_scores_external = model_v(model_w(input_external))
            binary_weight_external = F.softmax(binary_scores_external, 1)
            right_loss = - binary_weight_external[:, 1] * right_loss
            loss_v = left_loss + right_loss.mean()
            loss_v.backward()
            nn.utils.clip_grad_norm_(
                model_v.parameters(), args.grad_clip)
            optimizer_v.step()
        #architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.module.parameters(), args.grad_clip)
        optimizer.step()

        # update the parameter of w and h in teacher.
        optimizer_w.zero_grad()
        optimizer_h.zero_grad()

        teacher_logits = model_h(model_w(input))
        left_loss = criterion(teacher_logits, target)

        teacher_features = model_w(input_external)
        teacher_logits_external = model_h(teacher_features)
        right_loss = F.cross_entropy(
            teacher_logits_external, target_external, reduction='none')
        binary_scores_external = model_v(teacher_features)
        binary_weight_external = F.softmax(binary_scores_external, 1)
        right_loss = args.weight_gamma * \
            binary_weight_external[:, 1] * right_loss
        loss = left_loss + right_loss.mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model_w.parameters(), args.grad_clip)
        nn.utils.clip_grad_norm_(model_h.parameters(), args.grad_clip)

        optimizer_w.step()
        optimizer_h.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f',
                         step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)
            with torch.no_grad():
                logits = model(input)
                loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step,
                             objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
