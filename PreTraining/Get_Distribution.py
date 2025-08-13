
import os, pickle, time, shutil
from os.path import join
from glob import glob

import torch
import torch.nn as nn
from Densenet import densenet121, densenet161, densenet169, densenet201
from Folder import ImageFolder
from Resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, wide_resnet50_2, wide_resnet101_2
from Efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
from torchvision.models import alexnet
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
from scipy.special import softmax
import argparse
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image

begin = time.time()
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('-c','--cnn',  default='d121', help='dataset dir')
parser.add_argument('-d','--dir',  default='BRACS_ROI', help='dataset dir')  # chest_xray, hosp
parser.add_argument('-s','--save_dir',  default='Bracs_Densenet12111', help='save_dir')
parser.add_argument('-m','--multiple', default=2, type=int, help='multiple of input size')
parser.add_argument('-g','--gpu',  default='1', help='set 0,1 to use multi gpu for example')
args = parser.parse_args()


# exp settings
cnn = args.cnn
datasets_dir = args.dir
exp_dir = "/Your Path/result_archive/{}".format(args.save_dir)
batch_size = 1
# os.makedirs(exp_dir, exist_ok=True)
# shutil.copy('get_distribution.py', join(exp_dir, 'get_distribution.py'))


# data settings
data_dir = join("/Your Path/", datasets_dir)
data_sets = ['train', 'val', 'test']
#data_sets = ['Validation','Test']
# nb_class = len(os.listdir(join(data_dir, data_sets[0])))
nb_class = 7
re_size = int(128 * args.multiple)
crop_size = 112 * args.multiple
chex=1

## Allow Large Images
Image.MAX_IMAGE_PIXELS = None

# CUDA setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu




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
print('===== {} ====='.format(args.save_dir))
model_zoo = {'r18':resnet18, 'r34':resnet34, 'r50':resnet50, 'r101':resnet101, 'r152':resnet152, 
              'd121':densenet121, 'd161':densenet161, 'd169':densenet169, 'd201':densenet201,
             # 'eb0':efficientnet_b0, 'eb1':efficientnet_b1, 'eb2':efficientnet_b2, 'eb3':efficientnet_b3,
             # 'eb4':efficientnet_b4, 'eb5':efficientnet_b5, 'eb6':efficientnet_b6,  'eb7':efficientnet_b7,
             'rx50':resnext50_32x4d, 'alex':alexnet, 'wrn50':wide_resnet50_2, 'wrn101':wide_resnet101_2}
net = model_zoo[cnn](pretrained=True)

if not args.save_dir.endswith('_pre'):
    if cnn.startswith("r"):
        net.fc = nn.Linear(net.fc.in_features, nb_class)
    elif cnn.startswith('w'):
        net.fc = nn.Linear(net.fc.in_features, nb_class)
    elif cnn.startswith('d'):
        net.classifier = nn.Linear(net.classifier.in_features, nb_class)
    elif cnn.startswith('e'):
        net.classifier._modules['1'] = nn.Linear(net.classifier._modules['1'].in_features, nb_class)
    elif cnn.startswith("a"):
        net.classifier._modules['6'] = nn.Linear(net.classifier._modules['6'].in_features, nb_class)
    print(join(exp_dir, "max_*"))
    #net_file_name = glob(join(exp_dir, "max_*"))[0]
    #net_file_name = glob(join(exp_dir, "max_acc_66.35.pth"))[0]
    net_file_name = '/Your Path/Bracs_Densenet121/max_f1.pth'
    try:
        try: net.load_state_dict(torch.load(net_file_name)['model'])
        except: net.load_state_dict(torch.load(net_file_name))
    except:
        net = torch.nn.DataParallel(net)
        try: net.load_state_dict(torch.load(net_file_name)['model'])
        except: net.load_state_dict(torch.load(net_file_name))

net.cuda()
net.eval()

scores = []
for data_set in data_sets:
    testset = ImageFolder(
        root=os.path.join(data_dir, data_set), transform=transform_test
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    distribution_x4 = []
    distribution_xc = []
    y_pred, y_true, y_score = [], [], []
    paths = []
    test_loss = correct = total = 0

    with torch.no_grad():

        # for _, (inputs, targets, path) in enumerate(tqdm(testloader)):
        # 
        #     inputs, targets = inputs.cuda(), targets.cuda()
        #     try:
        #         x4, xc = net(inputs)
        #         distribution_x4.extend(x4.cpu().tolist())
        #     except:
        #         xc = net(inputs)
       for _, (inputs, targets, path) in enumerate(tqdm(testloader, ncols=80)):
            if chex == 1:
                bs, n_crops, c, h, w = inputs.size()
                inputs = inputs.view(-1, c, h, w)

            inputs, targets = inputs.cuda(), targets.cuda()
            try:
                x4, xc = net(inputs)
                
            except:
                xc = net(inputs)

            if chex == 1: 
              xc = xc.squeeze().view(bs, n_crops, -1).mean(1)
              x4 = x4.squeeze().view(bs, n_crops, -1).mean(1)
            

            _, predicted = torch.max(xc.data, 1)
            y_score.extend([_ for _ in softmax(xc.data.cpu(), axis=1)])
            y_pred.extend(predicted.cpu().tolist())

            distribution_x4.extend(x4.cpu().tolist())
            distribution_xc.extend(xc.cpu().tolist())
            y_true.extend(targets.cpu().tolist())
            paths.extend(path)
    # print(y_pred)
    # print(y_score)

    test_acc = 100.0 * accuracy_score(y_true, y_pred)
    test_f1 = 100.0 * f1_score(y_true, y_pred, average = 'micro')
    test_recall = 100.0 * recall_score(y_true, y_pred, average = 'micro')
    test_precision = 100.0 * precision_score(y_true, y_pred, average = 'micro')
    
    random_state = np.random.RandomState(0)
    n_samples, n_features = inputs.shape
    n_classes = len(np.unique(targets.cpu()))
    inputs = np.concatenate([inputs, random_state.randn(n_samples, 200 * n_features)], axis=1)
    (
      X_train,
      X_test,
      y_train,
      y_test,
    ) = train_test_split(inputs, targets, test_size=0.5, stratify=y, random_state=0)
    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test)
    y_onehot_test.shape  # (n_samples, n_classes)  
    
    test_auc = 100.0 * roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
    fig, ax = plt.subplots(figsize=(6, 6))

    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )
    
    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
        color="navy",
        linestyle=":",
        linewidth=4,
    )
    
    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for class_id, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_score[:, class_id],
            name=f"ROC curve for {target_names[class_id]}",
            color=color,
            ax=ax,
        )
    
    plt.plot([0, 1], [0, 1], "k--", label="ROC curve for chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass")
    plt.legend()
    plt.show()
    #
    
    
    # scores.append('{:.2f}'.format(test_acc))


    # scores.append('{:.4f}'.format(test_auc))

    print("Dataset\tf1\tACC\tprecision\trecall\tAUC")
    print("{}\t{:.2f}\t{:.2f}\t{:.2f}\t\t{:.2f}\t{:.2f}".format(data_set, test_f1, test_acc, test_precision, test_recall, test_auc))

    
    # # === 保存 pkl===

    # 保存 csv
    pd.DataFrame(y_true).to_csv(join(exp_dir, "targets_{}.csv".format(data_set)), index=None, header=None)
    pd.DataFrame(y_pred).to_csv(join(exp_dir, "predictions_{}.csv".format(data_set)), index=None, header=None)
    pd.DataFrame(paths).to_csv(join(exp_dir, "paths_{}.csv".format(data_set)), index=None, header=None)
    pd.DataFrame(distribution_x4).to_csv(join(exp_dir, "distribution_x4_{}.csv".format(data_set)), index=None, header=None)
    pd.DataFrame(distribution_xc).to_csv(join(exp_dir, "distribution_xc_{}.csv".format(data_set)), index=None, header=None)
    # 保存 csv

#     print("\nDataset {}\t{:.2f}\t{:.2f}\t{:.2f}\t\t{:.2f}\t{:.2f}".format(data_set, test_acc, test_f1, test_precision, test_recall, test_auc))
# for score in scores: print(score, end='\t')
print('\n')
# print('\n{:.2f}s used\n'.format(time.time() - begin))