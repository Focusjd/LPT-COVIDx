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
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from covidxdataset import COVIDxDataset
from torch.autograd import Variable
from model_search import Network
from architect_ts import Architect
from teacher import *
from teacher_update import *

from torch.utils.tensorboard import SummaryWriter   

parser = argparse.ArgumentParser("LPT-covidx")
parser.add_argument('--data', type=str, default='../data',
                    help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--learning_rate', type=float,
                    default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float,
                    default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float,
                    default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float,
                    default=50, help='report frequency')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--epochs', type=int, default=50,
                    help='num of training epochs')
parser.add_argument('--init_channels', type=int,
                    default=6, help='num of init channels')
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
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float,
                    default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float,
                    default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true',
                    default=True, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float,
                    default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float,
                    default=1e-3, help='weight decay for arch encoding')

# new hyperparams.
parser.add_argument('--weight_gamma', type=float, default=1.0)
parser.add_argument('--weight_lambda', type=float, default=1.0)
parser.add_argument('--model_v_learning_rate', type=float, default=3e-4)
parser.add_argument('--model_v_weight_decay', type=float, default=1e-3)
parser.add_argument('--learning_rate_w', type=float, default=0.025)
parser.add_argument('--learning_rate_h', type=float, default=0.025)
parser.add_argument('--weight_decay_w', type=float, default=3e-4)
parser.add_argument('--weight_decay_h', type=float, default=3e-4)
parser.add_argument('--is_parallel', type=int, default=0)
parser.add_argument('--teacher_arch', type=str, default='18')
parser.add_argument('--is_cifar100', type=int, default=0)
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
writer = SummaryWriter(args.save)


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


CIFAR_CLASSES = 10
CIFAR100_CLASSES = 100
COVID19_CLASSES = 3

gpus = [int(i) for i in args.gpu.split(',')]

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  if not args.is_parallel:
    torch.cuda.set_device('cuda:'+str(args.gpu))
    logging.info('gpu device = %d' % int(args.gpu))
    print('device numer: ', torch.cuda.device_count())
  else:
    logging.info('gpu device = %s' % args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled = True
  torch.cuda.manual_seed(args.seed)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  model = Network(args.init_channels, COVID19_CLASSES, args.layers, criterion)

  model = nn.DataParallel(model).cuda('cuda:'+str(args.gpu))
  if args.teacher_arch == '18':
    teacher_w = resnet18()
    teacher_w = nn.DataParallel(teacher_w).cuda('cuda:'+str(args.gpu))
  elif args.teacher_arch == '34':
    teacher_w = resnet34()
    teacher_w = nn.DataParallel(teacher_w).cuda('cuda:'+str(args.gpu))
  elif args.teacher_arch == '50':
    teacher_w = resnet50()
    teacher_w = nn.DataParallel(teacher_w).cuda('cuda:'+str(args.gpu))
  elif args.teacher_arch == '101':
    teacher_w = resnet101()
    teacher_w = nn.DataParallel(teacher_w).cuda('cuda:'+str(args.gpu))


  if args.is_cifar100:
    teacher_h = nn.Linear(512 * teacher_w.block.expansion, CIFAR100_CLASSES).cuda('cuda:'+str(args.gpu))
  else:
    # 512 * teacher_w.block.expansion = 512
    teacher_h = nn.Linear(512, COVID19_CLASSES)
    teacher_h = nn.DataParallel(teacher_h).cuda('cuda:'+str(args.gpu))

  # teacher_v = nn.Linear(512 * teacher_w.block.expansion, 2)
  teacher_v = nn.Linear(512, 2)
  teacher_v = nn.DataParallel(teacher_v).cuda('cuda:'+str(args.gpu))
  
  if args.is_parallel:
    # gpus = [int(i) for i in args.gpu.split(',')]
    # model = nn.DataParallel(
    #     model, device_ids=gpus)
    # teacher_w = nn.DataParallel(
    #     teacher_w, device_ids=gpus)
    # teacher_h = nn.DataParallel(
    #     teacher_h, device_ids=gpus)
    # teacher_v = nn.DataParallel(
    #     teacher_v, device_ids=gpus)
    model = model.module
    teacher_w = teacher_w.module
    teacher_h = teacher_h.module
    teacher_v = teacher_v.module

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)
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


  train_data = COVIDxDataset(mode='train', data_path=args.data)
  valid_data = COVIDxDataset(mode='validate', data_path=args.data)

  train_queue = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=2)
  valid_queue = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=False, num_workers=2)

  external_queue = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, pin_memory=False, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer, float(args.epochs), eta_min=args.learning_rate_min)
  scheduler_w = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer_w, float(args.epochs), eta_min=args.learning_rate_min)
  scheduler_h = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer_h, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)
  teacher_updater = Teacher_Updater(teacher_w, teacher_h, teacher_v, args)

  for epoch in range(args.epochs):
    lr = scheduler.get_lr()[0]
    lr_w = scheduler_w.get_lr()[0]
    lr_h = scheduler_h.get_lr()[0]
    logging.info('epoch %d lr %e lr_w %e lr_h %e', epoch, lr, lr_w, lr_h)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    # training
    train_acc, train_obj = train(
        train_queue, valid_queue, external_queue,
        model, architect, criterion, optimizer,
        optimizer_w,
        optimizer_h,
        lr,
        lr_w, lr_h,
        teacher_updater,
        teacher_w, teacher_h, teacher_v)
    logging.info('train_acc %f', train_acc)

    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Loss/train', train_obj, epoch)


    scheduler.step()
    scheduler_w.step()
    scheduler_h.step()
    # validation
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    writer.add_scalar('Accuracy/validate', valid_acc, epoch)
    writer.add_scalar('Loss/validate', valid_obj, epoch)

    # external_acc, external_obj = infer(external_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)
    # logging.info('external_acc %f', external_acc)

    utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, valid_queue, external_queue,
          model, architect, criterion, optimizer,
          optimizer_w,
          optimizer_h,
          lr,
          lr_w, lr_h,
          teacher_updater,
          teacher_w, teacher_h, teacher_v):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)

    input = input.cuda()
    target = target.cuda()

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = input_search.cuda()
    target_search = target_search.cuda()

    input_external, target_external = next(iter(external_queue))
    input_external = input_external.cuda()
    target_external = target_external.cuda()

    architect.step(input, target, input_external, target_external,
                   lr, optimizer, teacher_w, teacher_v, unrolled=args.unrolled)

    teacher_updater.step(criterion, input, target, input_search, target_search,
                         input_external, target_external,
                         lr_w, lr_h, optimizer, optimizer_w, optimizer_h,
                         model,
                         unrolled=args.unrolled)

    # update the parameter of w and h in teacher.
    optimizer_w.zero_grad()
    optimizer_h.zero_grad()

    teacher_logits = teacher_h(teacher_w(input))
    left_loss = criterion(teacher_logits, target)

    teacher_features = teacher_w(input_external)
    teacher_logits_external = teacher_h(teacher_features)
    right_loss = F.cross_entropy(
        teacher_logits_external, target_external, reduction='none')
    binary_scores_external = teacher_v(teacher_features)
    binary_weight_external = F.softmax(binary_scores_external, 1)
    right_loss = args.weight_gamma * \
        binary_weight_external[:, 1] * right_loss
    loss = left_loss + right_loss.mean()

    loss.backward()

    optimizer_w.step()
    optimizer_h.step()

    # update the model parameter.
    optimizer.zero_grad()
    logits = model(input)
    print("logits device: ", logits.device)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 3))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    # top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      # logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
      logging.info('train %03d %e %f', step, objs.avg, top1.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()
  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda()

        logits = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 3))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        # top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
          logging.info('valid %03d %e %f', step, objs.avg, top1.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main()

