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
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from covidxdataset import COVIDxDataset

from torch.autograd import Variable
from model import NetworkCIFAR as Network


parser = argparse.ArgumentParser("COVIDx")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=24, help='num of init channels')
parser.add_argument('--layers', type=int, default=16, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')

#darts-30epochs 
# parser.add_argument('--arch', type=str, default="Genotype(normal=[('avg_pool_3x3', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 3), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('skip_connect', 0), ('skip_connect', 3), ('skip_connect', 4), ('skip_connect', 3)], reduce_concat=range(2, 6))", help='which architecture to use')


# darts
# parser.add_argument('--arch', type=str, default="Genotype(normal=[('skip_connect', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 4), ('avg_pool_3x3', 0)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 3), ('skip_connect', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 2)], reduce_concat=range(2, 6))", help='which architecture to use')
# lpt
# parser.add_argument('--arch', type=str, default="Genotype(normal=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('dil_conv_3x3', 3), ('dil_conv_3x3', 4), ('dil_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 3), ('sep_conv_3x3', 0), ('max_pool_3x3', 3), ('max_pool_3x3', 4)], reduce_concat=range(2, 6))", help='which architecture to use')
# lpt 0.8 param
# parser.add_argument('--arch', type=str, default="Genotype(normal=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 2), ('skip_connect', 0), ('sep_conv_5x5', 3), ('max_pool_3x3', 0), ('dil_conv_3x3', 4), ('sep_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2), ('skip_connect', 0), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))", help='which architecture to use')

# lpt 8_layers 10 chennels
# parser.add_argument('--arch', type=str, default="Genotype(normal=[('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('sep_conv_5x5', 2), ('dil_conv_5x5', 3), ('sep_conv_5x5', 4), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 2), ('skip_connect', 3), ('avg_pool_3x3', 0), ('max_pool_3x3', 4), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))", help='which architecture to use')

# lpt 0.5-0.5 param
# parser.add_argument('--arch', type=str, default="Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('dil_conv_3x3', 3), ('dil_conv_5x5', 2), ('sep_conv_3x3', 2), ('dil_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 2), ('max_pool_3x3', 0), ('sep_conv_5x5', 2), ('dil_conv_5x5', 2), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))", help='which architecture to use')

# lpt 0.5-1 param
# parser.add_argument('--arch', type=str, default="Genotype(normal=[('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 2), ('max_pool_3x3', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 4), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))", help='which architecture to use')

# lpt 1-0.5 param
# parser.add_argument('--arch', type=str, default="Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 2), ('dil_conv_5x5', 1), ('sep_conv_5x5', 3), ('dil_conv_5x5', 2), ('sep_conv_3x3', 3), ('sep_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('dil_conv_5x5', 4), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))", help='which architecture to use')
 
# lpt 1-0.1 param    
parser.add_argument('--arch', type=str, default="Genotype(normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_3x3', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 4), ('sep_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 2), ('avg_pool_3x3', 0), ('sep_conv_5x5', 3), ('dil_conv_3x3', 0), ('dil_conv_5x5', 4), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))", help='which architecture to use')
    
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--is_cifar100', type=int, default=0)
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10
CIFAR100_CLASSES = 100
COVID19_CLASSES = 3


def save_checkpoint(state, checkpoint=args.save, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  genotype = eval("genotypes.%s" % args.arch)
  if args.is_cifar100:
    model = Network(args.init_channels, CIFAR100_CLASSES, args.layers, args.auxiliary, genotype)
  else:
    model = Network(args.init_channels, COVID19_CLASSES, args.layers, args.auxiliary, genotype)
  model = model.cuda()

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )
  if args.is_cifar100:
    train_transform, valid_transform = utils._data_transforms_cifar100(args)
  else:
    train_transform, valid_transform = utils._data_transforms_cifar10(args)
  if args.is_cifar100:
    train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
  else:
    train_data = COVIDxDataset(mode='eval_train', data_path=args.data)
    test_data = COVIDxDataset(mode='test', data_path=args.data)


  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=3)

  test_queue = torch.utils.data.DataLoader(
      test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=3)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
  start_epoch = 0
  if args.resume:
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
  for epoch in range(start_epoch, args.epochs):
    logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    logging.info('train_acc %f', train_acc)

    scheduler.step()

    test_acc, test_obj = infer(test_queue, model, criterion)
    logging.info('test_acc %f', test_acc)

    utils.save(model, os.path.join(args.save, 'weights.pt'))
    save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'scheduler': scheduler.state_dict(),
            'optimizer': optimizer.state_dict()})


def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    input = input.cuda()
    target = target.cuda(non_blocking=True)

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 3))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)

    if step % args.report_freq == 0:
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
        target = target.cuda(non_blocking=True)

        logits, _ = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 3))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)

        if step % args.report_freq == 0:
          logging.info('valid %03d %e %f', step, objs.avg, top1.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

