
import os, pickle
import argparse
import random
import shutil
from os.path import join
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
from Densenet import densenet121, densenet161, densenet169, densenet201
from Folder import ImageFolder
from Resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, wide_resnet50_2, wide_resnet101_2
from Efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
from Vgg import vgg11, vgg13, vgg16, vgg19
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from torch.utils.tensorboard import SummaryWriter
#from Focal_loss import focal_loss
from scipy.special import softmax
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.metrics import confusion_matrix
import sys
sys.path.append("..")
from roc_auc_score_multiclass import roc_auc_score_multiclass

#######################
##### 1 - Setting #####
#######################
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('-m','--multiple', default=2, type=int, help='multiple of input size')
parser.add_argument('--ckpt', default='', help='path of check_point model')
parser.add_argument('--requires_grad', default='all', help='the layers need finetune')
parser.add_argument('-d','--data_dir',  default='BRACS_ROI/', help='dataset dir')
parser.add_argument('-c','--cnn',  default='r50', help='CNN model')
parser.add_argument('-b','--batch_size',  default=32, type=int, help='batch_size')
parser.add_argument('--wt',  default=0, type=int, help='weight loss')
parser.add_argument('-g','--gpu',  default='0', help='gpu id')
parser.add_argument('--train_set', default='train', help='name of training set')
parser.add_argument('--test_set', default='val', help='name of testing set')
parser.add_argument('-w','--num_workers',  default=0, type=int, help='num_workers')
parser.add_argument('-e','--epoch',  default=200, type=int, help='epoch')
parser.add_argument('--chex',  default=1, type=int, help='use chexnet setting or not')
parser.add_argument('-r','--random_seed',  default=0, type=int, help='random seed')
parser.add_argument('-s','--save_dir',  default='Bracs_EfficientNetEB4', help='save_dir')
parser.add_argument('-l','--label_smooth',  default=0, type=float, help='label_smooth')
parser.add_argument('--learning_rate', default=0.001, type=float, help='learning_rate')
parser.add_argument('--scheduler', default=0, type=int, help='use scheduler')
parser.add_argument('-v','--evaluate', default=1, type=int, help='test every epoch')
parser.add_argument('-a', '--amp', default=2, type=int, help='0: w/o amp, 1: w/ nvidia apex.amp, 2: w/ torch.cuda.amp')
args = parser.parse_args()

chex = args.chex  # use chexnet setting
num_epoch = args.epoch
begin_epoch = 1
seed = args.random_seed
cnn = args.cnn
batch_size = args.batch_size
if args.learning_rate: lr_begin = args.learning_rate
else: lr_begin = (batch_size / 256) * 0.1
use_amp = args.amp
opt_level = "O1"
test_every_epoch = args.evaluate

## Allow Large Images
Image.MAX_IMAGE_PIXELS = None

# data settings
data_dir = join("your Path", args.data_dir)
# data_sets = ["hosp_val", 'hosp_test']
# data_sets = ["hosp_test", 'hosp_val']
data_sets = [args.train_set, args.test_set]
nb_class = len(os.listdir(join(data_dir, data_sets[0])))
re_size = int(128 * args.multiple)
crop_size = 112 * args.multiple
exp_dir = "your Path/result_archive/{}".format(args.save_dir)
summaryWriter = SummaryWriter(exp_dir)


# CUDA setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


# Random seed setting
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # multi gpu
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


# Dataloader
# hosp_val	([0.478, 0.478, 0.478], [0.276, 0.276, 0.276])
if chex == 0:
    train_transform = transforms.Compose(
        [
          
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=[0.5, 1.5]),  # brightness=[0.5, 1.5], contrast=[0.5, 1.5]
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # imagenet
            # transforms.Normalize([0.593, 0.593, 0.593], [0.191, 0.191, 0.191]),  # hosp
        ]
    )
else:
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transformList = []
    transformList.append(transforms.RandomResizedCrop(224))
    transformList.append(transforms.RandomHorizontalFlip())
    transformList.append(transforms.ToTensor())
    transformList.append(normalize)
    train_transform = transforms.Compose(transformList)

train_set = ImageFolder(root=join(data_dir, data_sets[0]), transform=train_transform)
#print(train_set)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=int(args.num_workers), drop_last=True
    # train_set, batch_size=batch_size, shuffle=True, num_workers=int(args.num_workers), drop_last=False
)

if chex == 0:
    transform_test = transforms.Compose(
        [
            transforms.Resize((re_size, re_size)),
            transforms.CenterCrop(crop_size),
            # transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # imagenet
            # transforms.Normalize([0.593, 0.593, 0.593], [0.191, 0.191, 0.191]),  # hosp
        ]
    )
else:
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transformList = []
    transformList.append(transforms.Resize(256))
    # transformList.append(transforms.Resize((256, 256)))
    # transformList.append(transforms.TenCrop(224))
    transformList.append(transforms.FiveCrop(224))
    transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
    transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
    transform_test = transforms.Compose(transformList)


# Net settings
model_zoo = {'r18':resnet18, 'r34':resnet34, 'r50':resnet50, 'r101':resnet101, 'r152':resnet152, 
              'd121':densenet121, 'd161':densenet161, 'd169':densenet169, 'd201':densenet201,
              'v11':vgg11, 'v13':vgg13, 'v16':vgg16, 'v19':vgg19,
             'eb0':efficientnet_b0, 'eb1':efficientnet_b1, 'eb2':efficientnet_b2, 'eb3':efficientnet_b3,
             'eb4':efficientnet_b4, 'eb5':efficientnet_b5, 'eb6':efficientnet_b6,  'eb7':efficientnet_b7,
             'rx50':resnext50_32x4d, 'wrn50':wide_resnet50_2, 'wrn101':wide_resnet101_2}  
net = model_zoo[cnn](pretrained=True)

if cnn.startswith("r"):
    net.fc = nn.Linear(net.fc.in_features, nb_class)
    # net.fc = nn.Sequential(nn.Dropout(p=args.drop_out), nn.Linear(net.fc.in_features, nb_class))  # for resnet
elif cnn.startswith('w'):
    net.fc = nn.Linear(net.fc.in_features, nb_class)
elif cnn.startswith('v'):
    net.classifier = nn.Linear(net.classifier[0].in_features, nb_class) # for VGG
elif cnn.startswith('d'):
    net.classifier = nn.Linear(net.classifier.in_features, nb_class)
    # net.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(net.classifier.in_features, nb_class))  # for densenet
elif cnn.startswith('e'):
    net.classifier._modules['1'] = nn.Linear(net.classifier._modules['1'].in_features, nb_class)
net.cuda()


# optimizer setting
# train_Loss = focal_loss(alpha=0.25)
if args.wt == 0: train_Loss = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smooth).cuda()) #, weight=torch.Tensor([0.19476485210143069, 0.09738242605071534, 0.1787430647820328, 0.11142796826956852, 0.17966680155093218, 0.10455797323339963, 0.13345691401192086]
else: train_Loss = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smooth, weight=torch.Tensor([0.19476485210143069, 0.09738242605071534, 0.1787430647820328, 0.11142796826956852, 0.17966680155093218, 0.10455797323339963, 0.13345691401192086]).cuda())
optimizer = torch.optim.SGD(net.parameters(), lr=lr_begin, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

if args.ckpt:
    print('=== use ckpt.pth ===')
    ckpt = torch.load(args.ckpt)
    net.load_state_dict(ckpt['model'])
   

if args.requires_grad == 'fc':
    for name, param in net.named_parameters():
        if "fc" in name: param.requires_grad = True
        else: param.requires_grad = False
elif args.requires_grad == 'all':
    for name, param in net.named_parameters():
        param.requires_grad = True


# Training
os.makedirs(exp_dir, exist_ok=True)
shutil.copyfile("train.sh", exp_dir + "/train.sh")
shutil.copyfile("train.py", exp_dir + "/train.py")
shutil.copyfile("Folder.py", exp_dir + "/Folder.py")
shutil.copyfile("Densenet.py", exp_dir + "/Densenet.py")
shutil.copyfile("Resnet.py", exp_dir + "/Resnet.py")


# Amp
if use_amp == 1:  # use nvidia apex.amp
    print('\n===== Using NVIDIA AMP =====')
    from apex import amp

    # net.cuda()
    net, optimizer = amp.initialize(net, optimizer, opt_level='O1')
    with open(os.path.join(exp_dir, 'train_log.csv'), 'a+') as file:
        file.write('===== Using NVIDIA AMP =====\n')
elif use_amp == 2:  # use torch.cuda.amp
    print('\n===== Using Torch AMP =====')
    from torch.cuda.amp import GradScaler, autocast

    scaler = GradScaler()
    with open(os.path.join(exp_dir, 'train_log.csv'), 'a+') as file:
        file.write('===== Using Torch AMP =====\n')

if len(args.gpu) > 1: net = torch.nn.DataParallel(net)


########################
##### 2 - Training #####
########################
min_train_loss = float('inf')
max_val_acc = 0

for epoch in range(begin_epoch, num_epoch + 1):
    # if epoch > 1: break
    print("\n===== Epoch: {} / {} =====".format(epoch, num_epoch))
    net.train()
    lr_now = optimizer.param_groups[0]["lr"]
    train_loss = 0
    y_pred, y_true, y_score = [], [], []

    for batch_idx, (inputs, targets, _) in enumerate(tqdm(train_loader, ncols=80)):
        optimizer.zero_grad()
        y_true.extend(targets)
        inputs, targets = inputs.cuda(), targets.cuda()
        #print(inputs)
        ##### amp setting
        if use_amp == 1:  # use nvidia apex.amp
            x4, xc = net(inputs)
            loss = train_Loss(xc, targets)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
        elif use_amp == 2:  # use torch.cuda.amp
            with autocast():
                x4, xc = net(inputs)
                loss = train_Loss(xc, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            x4, xc = net(inputs)
            loss = train_Loss(xc, targets)
            loss.backward()
            optimizer.step()
        _, predicted = torch.max(xc.data, 1)
        
        y_score.extend([_ for _ in softmax(xc.data.cpu(), axis=1)])
        y_pred.extend(predicted.data.cpu())
        
        train_loss += loss.item()
    
    
    #print(type(y_true))
    if args.scheduler == 1: scheduler.step()
    train_loss /= len(y_true)
    # train_loss /= len(train_loader)
    train_f1 = 100 * f1_score(y_true, y_pred,  average='weighted')
    train_recall = 100 * recall_score(y_true, y_pred,  average='weighted')
    train_precision = 100 * precision_score(y_true, y_pred,  average='weighted')
    train_acc = 100 * accuracy_score(y_true, y_pred)
    
    print(
        "Train | lr: {:.4f} | Loss: {:.4f} | Acc: {:.3f} | F1: {:.3f}".format(
            lr_now, train_loss, train_acc, train_f1
        )
    )
    
    # Evaluating
    if test_every_epoch == 0:
        # Save last epoch model
        torch.save({'epoch': epoch,
                    'model': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    },
                   os.path.join(exp_dir, "ckpt.pth"),
                   _use_new_zipfile_serialization=False)

    else:
        net.eval()
        val_Loss = torch.nn.CrossEntropyLoss(weight=torch.Tensor([0.13775589304958652, 0.14736676930885997, 0.13775589304958652, 0.12932185878124447, 0.1545553922019751, 0.1584192770070245, 0.13482491660172297]).cuda())
        val_loss = 0
        with torch.no_grad():
            val_set = ImageFolder(root=join(data_dir, data_sets[-1]), transform=transform_test)
            #print(val_set)
            val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
            y_pred, y_true, y_score = [], [], []

            for _, (inputs, targets, _) in enumerate(tqdm(val_loader, ncols=80)):
                if chex == 1:
                    bs, n_crops, c, h, w = inputs.size()
                    inputs = inputs.view(-1, c, h, w)
                y_true.extend(targets)
                inputs, targets = inputs.cuda(), targets.cuda()
                #print(inputs)
                try: x4, xc = net(inputs)
                except: xc = net(inputs)
                if chex == 1: xc = xc.squeeze().view(bs, n_crops, -1).mean(1)
                
                _, predicted = torch.max(xc.data, 1)
                y_score.extend([_ for _ in softmax(xc.data.cpu(), axis=1)])
                y_pred.extend(predicted.data.cpu())
                loss = val_Loss(xc, targets)
                val_loss += loss.item()

            val_loss /= len(y_true)
            val_f1 = 100 * f1_score(y_true, y_pred, average='weighted')
            val_recall = 100 * recall_score(y_true, y_pred, average='weighted')
            val_precision = 100 * precision_score(y_true, y_pred, average='weighted')
            val_acc = 100 * accuracy_score(y_true, y_pred)
            #val_auc = 100.0 * roc_auc_score(y_true, y_score,   multi_class="ovr", average="micro")
            #print('{} | Loss: {:.4f} | Acc: {:.3f} | F1: {:.3f} | Auc: {:.2f}'.format(data_sets[-1], val_loss, val_acc, val_f1, val_auc))
            print('{} | Loss: {:.4f} | Acc: {:.3f} | F1: {:.3f} | '.format(data_sets[-1], val_loss, val_acc, val_f1))
            summaryWriter.add_scalars(main_tag='', tag_scalar_dict={"train_loss":train_loss,
                                                                    'train_recall': train_recall,
                                                                    'train_precision': train_precision,
                                                                    "train_f1":train_f1,
                                                                    "train_acc": train_acc,
                                                                    #"train_auc": train_auc,
                                                                    "val_loss":val_loss,
                                                                    'val_recall':val_recall,
                                                                    'val_precision':val_precision,
                                                                    "val_f1":val_f1,
                                                                    "val_acc": val_acc,
                                                                    #"val_auc": val_auc,
                                                                    'lr':lr_now}, global_step=epoch)
            patience = 10  # Number of epochs without improvement
            epochs_since_improvement = 0
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                epochs_since_improvement = 0
                old_models = sorted(glob(join(exp_dir, 'max_*.pth')))
                if len(old_models) > 0: os.remove(old_models[0])
                torch.save({'epoch': epoch,
                            'model': net.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            },
                           os.path.join(exp_dir, "max_acc_{:.2f}.pth".format(max_val_acc)),
                           _use_new_zipfile_serialization=False)
            else:
                epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print(f'Early stopping: No improvement in {patience} epochs.')
                break




########################
##### 3 - Testing  #####
########################
print("\n\n===== TESTING =====")
print(args.save_dir)
# reload the model for testing
net = model_zoo[cnn]()

if cnn.startswith("r"):
    net.fc = nn.Linear(net.fc.in_features, nb_class)
    # net.fc = nn.Sequential(nn.Dropout(p=args.drop_out), nn.Linear(net.fc.in_features, nb_class))  # for resnet
elif cnn.startswith('w'):
    net.fc = nn.Linear(net.fc.in_features, nb_class) # for wideresnet
elif cnn.startswith('v'):
    net.classifier = nn.Linear(net.classifier[0].in_features, nb_class) # for VGG
elif cnn.startswith('d'):
    net.classifier = nn.Linear(net.classifier.in_features, nb_class)
    # net.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(net.classifier.in_features, nb_class))  # for densenet
elif cnn.startswith('e'):
    net.classifier._modules['1'] = nn.Linear(net.classifier._modules['1'].in_features, nb_class)
    # net.classifier = nn.Linear(net.classifier._modules['1'].in_features, nb_class)

if len(args.gpu) > 1: net = torch.nn.DataParallel(net)
net_file_name = glob(join(exp_dir, "max_*.pth"))[0]
net.load_state_dict(torch.load(join(exp_dir, net_file_name))['model'])
net.eval()
net.cuda()


#for data_set in data_sets:
for data_set in data_sets:
    testset = ImageFolder(
        root=os.path.join(data_dir, 'test'), transform=transform_test
    )
    #print(testset)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    distribution_x4 = []
    distribution_xc = []
    paths = []
    test_loss = torch.nn.CrossEntropyLoss(weight=torch.Tensor([0.14352798073922118, 0.1471616005047711, 0.14177763951069408, 0.14006947517923993, 0.1471616005047711, 0.13677372282208136, 0.14352798073922118]).cuda())
    y_pred, y_true, y_score = [], [], []

    with torch.no_grad():

        for _, (inputs, targets, path) in enumerate(tqdm(testloader, ncols=80)):
            if chex == 1:
                bs, n_crops, c, h, w = inputs.size()
                inputs = inputs.view(-1, c, h, w)

            inputs, targets = inputs.cuda(), targets.cuda()
            try:
                x4, xc = net(inputs)
                distribution_x4.extend(x4.cpu().tolist())
            except:
                xc = net(inputs)

            if chex == 1: xc = xc.squeeze().view(bs, n_crops, -1).mean(1)

            _, predicted = torch.max(xc.data, 1)
            y_score.extend([_ for _ in softmax(xc.data.cpu(), axis=1)])
            y_pred.extend(predicted.cpu().tolist())

            distribution_x4.extend(x4.cpu().tolist())
            distribution_xc.extend(xc.cpu().tolist())
            y_true.extend(targets.cpu().tolist())
            paths.extend(path)

    test_acc = 100.0 * accuracy_score(y_true, y_pred)
    test_f1 = 100.0 * f1_score(y_true, y_pred, average='weighted')
    test_recall = 100.0 * recall_score(y_true, y_pred, average='weighted')
    test_precision = 100.0 * precision_score(y_true, y_pred, average='weighted')
    #test_auc = 100.0 * roc_auc_score(y_true, y_score,   multi_class="ovr", average="micro")
    #print("Dataset {}\t{:.2f}\t{:.2f}\t{:.2f}\t\t{:.2f}\t{:.2f}\n".format(data_set, test_acc, test_f1, test_precision, test_recall, test_auc))
    print("Dataset {}\t{:.2f}\t{:.2f}\t{:.2f}\t\t{:.2f}\t".format('Test Set', test_acc, test_f1, test_precision, test_recall))
    #Logging
    with open(os.path.join(exp_dir, "{:.2f}_{}\n".format(test_f1, data_set)), "a+") as file:
        pass
    pickle.dump(y_true, open(join(exp_dir, "targets_{}.pkl".format(data_set)), 'wb+'))
    pickle.dump(y_pred, open(join(exp_dir, "predictions_{}.pkl".format(data_set)), 'wb+'))
    pickle.dump(paths, open(join(exp_dir, "paths_{}.pkl".format(data_set)), 'wb+'))
    pickle.dump(distribution_x4, open(join(exp_dir, "distribution_x4_{}.pkl".format(data_set)), 'wb+'))
    pickle.dump(distribution_xc, open(join(exp_dir, "distribution_xc_{}.pkl".format(data_set)), 'wb+'))

