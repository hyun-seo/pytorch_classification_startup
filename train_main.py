from utils.accuracy import accuracy
from models.resnet import resnet18
import os
from utils.meters import Meter
import torch
from tqdm import tqdm
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from params import Params


params = Params()
params.build()

torch.cuda.manual_seed_all(params.seed)


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
if model == None:
    raise NotImplementedError

model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=params.lr,
                      momentum=params.momentum, weight_decay=params.weight_decay)

state = dict()


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in params.schedule:
        state['lr'] *= params.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


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
    adjust_learning_rate(optimizer, e)

    print(f"===== epoch : {e} =====")

    model.train()
    train_loss.reset()
    train_top1.reset()
    train_top5.reset()
    for input in tqdm(trainloader):
        inputs, targets = input
        inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        train_loss.update(loss.item(), inputs.size(0))
        _, predicted = outputs.max(1)
        train_top1.update(predicted.eq(targets).sum().item(), inputs.size(0))



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
            _, predicted = outputs.max(1)
            test_top1.update(predicted.eq(targets).sum().item(), inputs.size(0))

        logger.add_scalar("test/loss", test_loss.avg(), e)
        logger.add_scalar("test/train_top1", test_top1.avg(), e)

    torch.save(model.state_dict(), os.path.join(params.save_dir, "model.pth"))
