import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torchvision.utils import save_image
from utils.resnet import resnet50
from utils.MODE import MODE_A, MODE_F

from utils.datasets import ImageFolder, ConcatDataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data_path',
                    type=str,
                    default='',
                    help='path to dataset')
parser.add_argument('-j',
                    '--workers',
                    default=12,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs',
                    default=50,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b',
                    '--batch-size',
                    default=256,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 3200), this is the total '
                    'batch size of all GPUs on the current node when '
                    'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=2e-4,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    metavar='M',
                    help='momentum')
parser.add_argument('--local_rank',
                    default=-1,
                    type=int,
                    help='node rank for distributed training')
parser.add_argument('--wd',
                    '--weight-decay',
                    default=1e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p',
                    '--print-freq',
                    default=10,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('-e',
                    '--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed',
                    default=None,
                    type=int,
                    help='seed for initializing training. ')

parser.add_argument("--mode", default="F",
                    type=str, help="Fourier or AdaIN")

parser.add_argument("--beta", default = 0.4, type=float)

parser.add_argument("--test_index", default = 5, type=int)


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def main():
    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()
    if args.local_rank == 0:
        print("Num of GPU:", args.nprocs)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    main_worker(args.local_rank, args.nprocs, args)


def main_worker(local_rank, nprocs, args):
    best_acc_val = .0
    best_acc_test = .0
    num_classes = 345
    num_classes = 345
    dist.init_process_group(backend='nccl')
    # create model
    model = resnet50(pretrained=False, num_classes=num_classes)
    weight = torch.load(args.data_path+"/resnet50-0676ba61.pth")
    weight['fc.weight'] = model.state_dict()['fc.weight']
    weight['fc.bias'] = model.state_dict()['fc.bias']
    model.load_state_dict(weight)
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    args.batch_size = int(args.batch_size / nprocs)
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank])
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(local_rank)
    
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    
    if args.mode == "A":
        mode = MODE_A(
            decoder_weights="./decoder.pth",
            vgg_weights="./vgg_normalised.pth",
            device=local_rank,
            gamma=1,
            num_mix=6,
            move_step=5,
            mu=0.05,
            criterion=criterion,
            norm_mean=norm_mean,
            norm_std=norm_std
        )
    else:
        mode = MODE_F(
            device=local_rank,
            gamma=1,
            num_mix=12,
            move_step=7,
            mu=0.05,
            criterion=criterion,
        )


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    cudnn.benchmark = True

    # Data loading code

    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]


    train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(norm_mean, norm_std)])

    test_transform = transforms.Compose([transforms.Resize((256,256)),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(norm_mean, norm_std)])

    source_root = args.data_path+'/domainnet'
    train_name = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
    # train_name = ["clipart", "painting", "real", "sketch"]
    test_name = train_name.pop(args.test_index)
    
    train_datalists_root = []
    for i in range(len(train_name)):
        train_datalists_root.append(
            args.data_path+"/domainnet/splits/{}_train.txt".format(train_name[i]))
    
    val_datalists_root = []
    for i in range(len(train_name)):
        val_datalists_root.append(
            args.data_path+"/domainnet/splits/{}_test.txt".format(train_name[i]))

    test_datalists_root = args.data_path+"/domainnet/splits/{}_test.txt".format(test_name)
    
    train_set = []
    for i in range(len(train_name)):
        train_set.append(ImageFolder(
            root=args.data_path+'/domainnet/{}'.format(train_name[i]), transform=train_transform, datalists_path=train_datalists_root[i], source_root=source_root))

    train_dataset = ConcatDataset(train_set)

    
    val_set = []
    for i in range(len(train_name)):
        val_set.append(ImageFolder(
            root=args.data_path+'/domainnet/{}'.format(train_name[i]), transform=test_transform, datalists_path=val_datalists_root[i], source_root=source_root))

    val_dataset = ConcatDataset(val_set)

    
    test_dataset = ImageFolder(
        root=args.data_path+'/domainnet/{}'.format(test_name), transform=test_transform, datalists_path=test_datalists_root, source_root=source_root)


    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               sampler=train_sampler)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             pin_memory=False,
                                             sampler=val_sampler)

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             pin_memory=False,
                                             sampler=test_sampler)


    if args.evaluate:
        validate(val_loader, model, criterion, local_rank, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, mode, criterion, optimizer, epoch, local_rank,
              args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, local_rank, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc_val
        best_acc_val = max(acc1, best_acc_val)
        
        if is_best and args.local_rank == 0:
            print("######")
            print("Best val acc : {} in epoch {}".format(best_acc_val, epoch))
            print("######")
        
        # evaluate on test set
        acc2 = test(test_loader, model, criterion, local_rank, args)

        # remember best acc@1 and save checkpoint
        is_best = acc2 > best_acc_test
        best_acc_test = max(acc2, best_acc_test)
        
        if is_best and args.local_rank == 0:
            print("######")
            print("Best test acc : {} in epoch {}".format(best_acc_test, epoch))
            print("######")
        
        if args.local_rank == 0:
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.module.state_dict(),
                    'best_acc1': best_acc_val,
                }, is_best, model.module.state_dict())


def train(train_loader, model, mode, criterion, optimizer, epoch, local_rank, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))



    end = time.time()
    for i, ((x, label), domain) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        save_flag = (i % (40 * args.print_freq) == 0)
        
        x = x.cuda(local_rank, non_blocking=True)
        label = label.cuda(local_rank, non_blocking=True)
        domain = domain.cuda(local_rank, non_blocking=True)

        # images = images.cuda(local_rank, non_blocking=True)
        # target = target.cuda(local_rank, non_blocking=True)
        
        if args.mode == "None":
            # switch to train mode
            model.train()
            # compute output
            score_real = model(x)
            Loss_total = criterion(score_real, label)
            
            if save_flag:
                print("#######")
                print()
                print('Task epoch {}'.format(epoch))
                print()
                print('Loss_real:{}'.format(Loss_total.item()))
                print()
                print("#######")
        else:
            model.eval()
            x_final, log = mode.main(x.clone(), label, model, domain, save_flag=save_flag)
            
            # switch to train mode
            model.train()
            # compute output
            input = torch.cat([x, x_final], dim=0).detach()
            score = model(input)
            score_real, score_final = torch.chunk(score, 2, dim=0)
            Loss_real = criterion(score_real, label)
            Loss_final = criterion(score_final, label)
            Loss_total = (1 - args.beta) * Loss_real + args.beta * Loss_final

        # measure accuracy and record loss
        acc1, acc5 = accuracy(score_real, label, topk=(1, 5))

        torch.distributed.barrier()

        reduced_loss = reduce_mean(Loss_total, args.nprocs)
        reduced_acc1 = reduce_mean(acc1, args.nprocs)
        reduced_acc5 = reduce_mean(acc5, args.nprocs)

        losses.update(reduced_loss.item(), x.size(0))
        top1.update(reduced_acc1.item(), x.size(0))
        top5.update(reduced_acc5.item(), x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        Loss_total.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.local_rank == (i % 4):
            progress.display(i)
        
        if save_flag and args.local_rank == 0 and args.mode != "None":
            print("Loss real:", Loss_real.item())
            print("Loss final:", Loss_final.item())
        
        if save_flag and args.local_rank == 0 and args.mode != "None":
            img_temp = torch.cat(torch.chunk(log["save_img_temp"], args.batch_size, dim=0), dim=1).squeeze(0)
            save_image(denormalize(img_temp, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                '/output/Epoch_{}_{}.jpg'.format(epoch, i), nrow = 7 if args.mode=="AdaIN" else 9)

def validate(val_loader, model, criterion, local_rank, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, ((images, target), domain) in enumerate(val_loader):
            images = images.cuda(local_rank, non_blocking=True)
            target = target.cuda(local_rank, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            torch.distributed.barrier()

            reduced_loss = reduce_mean(loss, args.nprocs)
            reduced_acc1 = reduce_mean(acc1, args.nprocs)
            reduced_acc5 = reduce_mean(acc5, args.nprocs)

            losses.update(reduced_loss.item(), images.size(0))
            top1.update(reduced_acc1.item(), images.size(0))
            top5.update(reduced_acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1,
                                                                    top5=top5))

    return top1.avg

def test(test_loader, model, criterion, local_rank, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(test_loader), [batch_time, losses, top1, top5],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(test_loader):
            images = images.cuda(local_rank, non_blocking=True)
            target = target.cuda(local_rank, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            torch.distributed.barrier()

            reduced_loss = reduce_mean(loss, args.nprocs)
            reduced_acc1 = reduce_mean(acc1, args.nprocs)
            reduced_acc5 = reduce_mean(acc5, args.nprocs)

            losses.update(reduced_loss.item(), images.size(0))
            top1.update(reduced_acc1.item(), images.size(0))
            top5.update(reduced_acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1,
                                                                    top5=top5))

    return top1.avg

def save_checkpoint(state, is_best, model_sd, filename='checkpoint.pth.tar'):
    # torch.save(state, filename)
    if is_best:
        # shutil.copyfile(filename, 'model_best.pth.tar')
        torch.save(model_sd, f="/output/BEST.pth")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def denormalize(x, mean=None, std=None):
    mean = [0.485, 0.456, 0.406] if mean is None else mean
    std = [0.229, 0.224, 0.225] if std is None else std
    mean = torch.tensor(mean).cuda()
    std = torch.tensor(std).cuda()
    x = x * std.view(1, 3, 1, 1)
    x += mean.view(1, 3, 1, 1)
    return x

if __name__ == '__main__':
    main()


