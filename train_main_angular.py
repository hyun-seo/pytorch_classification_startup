import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from tqdm import tqdm
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from models.angular import AngleLoss, resnet18_angular, resnet34_angular

from models.resnet import resnet18, resnet34
from utils.accuracy import accuracy
from utils.meters import Meter
from params import Params
import random
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default="debug")
parser.add_argument("--model", type=str, default="resnet18_angular")
parser.add_argument("--dataset", type=str, default="cifar100")
parser.add_argument("--drop", type=str, default="False")
parser.add_argument("--a", type=float, default="0.5")
parser.add_argument("--p", type=float, default="0.5")

args = parser.parse_args()
# args = parser.parse_args("--model resnet18_randdrop --drop True".split())

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

params = Params()
params.exp_name = args.exp_name
params.model = args.model
params.dataset = args.dataset
params.drop = args.drop
params.a = args.a
params.p = args.p
params.build()
print(params)

torch.cuda.manual_seed_all(params.seed)
random.seed(params.seed)
np.random.seed(params.seed)
torch.manual_seed(params.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)  # if use multi-GPU
# It could be slow
cudnn_ok = True
if cudnn_ok:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =======================================
# Data
print('==> Preparing dataset %s' % params.dataset)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


if params.dataset == 'cifar10':
    dataloader = datasets.CIFAR10
    num_classes = 10
else:
    dataloader = datasets.CIFAR100
    num_classes = 100


trainset = dataloader(root='./data', train=True,
                      download=True, transform=transform_train)
trainloader = data.DataLoader(
    trainset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_worker)

testset = dataloader(root='./data', train=False,
                     download=False, transform=transform_test)
testloader = data.DataLoader(
    testset, batch_size=params.batch_size, shuffle=False, num_workers=params.num_worker)


# =======================================
# Model
model = None
if params.model == "resnet18":
    model = resnet18(num_classes)
elif params.model == "resnet34":
    model = resnet34(num_classes)
elif params.model == "resnet18_angular":
    model = resnet18_angular()
elif params.model == "resnet34_angular":
    model = resnet34_angular()
if model == None:
    raise NotImplementedError


if torch.cuda.is_available():
    model.cuda()
# criterion = nn.CrossEntropyLoss()
# criterion_kl = nn.KLDivLoss(reduction='batchmean')

criterion = AngleLoss()

optimizer = optim.SGD(model.parameters(), lr=params.lr,
                      momentum=params.momentum, weight_decay=params.weight_decay)
scheduler = MultiStepLR(optimizer, milestones=params.schedule, gamma=0.1)

state = dict()
state['lr'] = 0.1


# =======================================
# logger
logger = SummaryWriter(log_dir=params.tb_dir)


train_loss = Meter()
train_top1 = Meter()
train_top5 = Meter()

test_loss = Meter()
test_top1 = Meter()
test_top5 = Meter()


# =======================================


for e in range(params.epoch):
    print(f"===== epoch : {e} =====")

    model.train()
    model.parameters()
    train_loss.reset()
    train_top1.reset()
    train_top5.reset()
    for input in tqdm(trainloader):
        inputs, targets = input
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()

            # compute output
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            # 0=cos_theta 1=phi_theta
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure accuracy and record loss
        train_loss.update(loss.item(), inputs.size(0))
        _, predicted = outputs[0].max(1)
        train_top1.update(predicted.eq(targets).sum().item(), inputs.size(0))

    scheduler.step()

    logger.add_scalar("train/loss", train_loss.avg(), e)
    logger.add_scalar("train/train_top1", train_top1.avg(), e)

    model.eval()
    test_loss.reset()
    test_top1.reset()
    test_top5.reset()
    with torch.no_grad():
        for input in tqdm(testloader):
            inputs, targets = input
            inputs, targets = inputs.cuda(), targets.cuda()

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            test_loss.update(loss.item(), inputs.size(0))
            _, predicted = outputs[0].max(1)
            test_top1.update(predicted.eq(
                targets).sum().item(), inputs.size(0))
        print(test_top1.avg())
        logger.add_scalar("test/loss", test_loss.avg(), e)
        logger.add_scalar("test/train_top1", test_top1.avg(), e)

    torch.save(model.state_dict(), os.path.join(params.save_dir, "model.pth"))
