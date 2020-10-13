import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import torch.nn.functional as F
from cutout import Cutout
from models.resnet import *
from models.preactresnet import *
from models.senet import *
from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Task-Oriented Feature Distillation. ')
parser.add_argument('--model', default="resnet18", help="choose the student model", type=str)
parser.add_argument('--dataset', default="cifar100", type=str, help="cifar10/cifar100")
parser.add_argument('--alpha', default=0.05, type=float)
parser.add_argument('--beta', default=0.03, type=float)
parser.add_argument('--l2', default=7e-3, type=float)
parser.add_argument('--teacher', default="resnet18", type=str)
parser.add_argument('--t', default=3.0, type=float, help="temperature for logit distillation ")
args = parser.parse_args()
print(args)

BATCH_SIZE = 128
LR = 0.1

transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4, fill=128),
                         transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                         Cutout(n_holes=1, length=16),
                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset, testset = None, None
if args.dataset == 'cifar100':
    trainset = torchvision.datasets.CIFAR100(
        root='data',
        train=True,
        download=True,
        transform=transform_train
    )
    testset = torchvision.datasets.CIFAR100(
        root='data',
        train=False,
        download=True,
        transform=transform_test
    )
if args.dataset == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(
        root='data',
        train=True,
        download=True,
        transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root='data',
        train=False,
        download=True,
        transform=transform_test
    )
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)

#   get the student model
if args.model == "resnet18":
    net = resnet18()
if args.model == "resnet50":
    net = resnet50()
if args.model == "resnet101":
    net = resnet101()
if args.model == "resnet152":
    net = resnet152()
if args.model == "resnext50":
    net = resnext50_32x4d()
if args.model == "mobilenet":
    net = mobilenet()
if args.model == "mobilenetv2":
    net = mobilenetv2()
if args.model == "shufflenet":
    net = shufflenet()
if args.model == "shufflenetv2":
    net = shufflenetv2()
if args.model == "preactresnet18":
    net = preactresnet18()
    LR = 0.02
    # reduce init lr for stable training
if args.model == "preactresnet50":
    net = preactresnet50()
    LR = 0.02
    # reduce init lr for stable training
if args.model == "senet18":
    net = seresnet18()
if args.model == "senet50":
    net = seresnet50()

#   get the teacher model
if args.teacher == 'resnet18':
    teacher = resnet18()
elif args.teacher == 'resnet50':
    teacher = resnet50()
elif args.teacher == 'resnet101':
    teacher = resnet101()
elif args.teacher == 'resnet152':
    teacher = resnet152()


teacher.load_state_dict(torch.load("./teacher/"+args.teacher+".pth"))
teacher.cuda()
net.to(device)
orthogonal_penalty = args.beta
init = False
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, weight_decay=args.l2, momentum=0.9)

if __name__ == "__main__":
    best_acc = 0
    print("Start Training")
    for epoch in range(250):
        if epoch in [80, 160, 240]:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(trainloader, 0):
            length = len(trainloader)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, student_feature = net(inputs)

            #   get teacher results
            with torch.no_grad():
                teacher_logits, teacher_feature = teacher(inputs)

            #   init the feature resizing layer depending on the feature size of students and teachers
            #   a fully connected layer is used as feature resizing layer here
            if not init:
                teacher_feature_size = teacher_feature[0].size(1)
                student_feature_size = student_feature[0].size(1)
                num_auxiliary_classifier = len(teacher_logits)
                link = []
                for j in range(num_auxiliary_classifier):
                    link.append(nn.Linear(student_feature_size, teacher_feature_size, bias=False))
                net.link = nn.ModuleList(link)
                net.cuda()
                #   we redefine optimizer here so it can optimize the net.link layers.
                optimizer = optim.SGD(net.parameters(), lr=LR, weight_decay=5e-4, momentum=0.9)
                init = True

            #   compute loss
            loss = torch.FloatTensor([0.]).to(device)

            #   Distillation Loss + Task Loss
            for index in range(len(student_feature)):
                student_feature[index] = net.link[index](student_feature[index])
                #   task-oriented feature distillation loss
                loss += torch.dist(student_feature[index], teacher_feature[index], p=2) * args.alpha
                #   task loss (cross entropy loss for the classification task)
                loss += criterion(outputs[index], labels)
                #   logit distillation loss, CrossEntropy implemented in utils.py.
                loss += CrossEntropy(outputs[index], teacher_logits[index], 1 + (args.t/250) * float(1+epoch))

            # Orthogonal Loss
            for index in range(len(student_feature)):
                weight = list(net.link[index].parameters())[0]
                weight_trans = weight.permute(1, 0)
                ones = torch.eye(weight.size(0)).cuda()
                ones2 = torch.eye(weight.size(1)).cuda()
                loss += torch.dist(torch.mm(weight, weight_trans), ones, p=2) * orthogonal_penalty
                loss += torch.dist(torch.mm(weight_trans, weight), ones2, p=2) * orthogonal_penalty

            sum_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(labels.size(0))
            _, predicted = torch.max(outputs[0].data, 1)
            correct += float(predicted.eq(labels.data).cpu().sum())

            if i % 20 == 0:
                print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.2f%% '
                      % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1),
                         100 * correct / total))

        print("Waiting Test!")
        with torch.no_grad():
            correct = 0.0
            total = 0.0
            for data in testloader:
                net.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs, feature = net(images)
                _, predicted = torch.max(outputs[0].data, 1)
                correct += float(predicted.eq(labels.data).cpu().sum())
                total += float(labels.size(0))

            print('Test Set AccuracyAcc:  %.4f%% ' % (100 * correct / total))
            if correct/total > best_acc:
                best_acc = correct/total
                print("Best Accuracy Updated: ", best_acc * 100)
                torch.save(net.state_dict(), "./checkpoint/"+args.model+".pth")
    print("Training Finished, Best Accuracy is %.4f%%" % (best_acc * 100))


