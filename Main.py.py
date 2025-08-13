import os
import numpy as np
import time
import sys

from AdaptiveTraining import ChexnetTrainer


#--------------------------------------------------------------------------------   

def runTrain():
    DENSENET121 = 'DENSE-NET-121'
    DENSENET169 = 'DENSE-NET-169'
    DENSENET201 = 'DENSE-NET-201'
    RSENET101='RESNET-101'
    RSENET50='RESNET-50'
    WIDERSENET50='WIDE-RESNET-50'
    EFFICIENTNETB4 = 'EFFICIENT-NET_B4'
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime

    # ---- Path to the directory with images
    pathImgTrain = '/Your Path/BRACS_ROI/train'
    pathImgVal = '/Your Path/BRACS_ROI/val'
    pathImgTest = '/Your Path/BRACS_ROI/test'

    # ---- Neural network parameters: type of the network, is it pre-trained
    # ---- on imagenet, number of classes
    nnArchitecture = DENSENET121
    nnIsTrained = True
    nnClassCount = 7  # 14

    # ---- Training settings: batch size, maximum number of epochs
    trBatchSize = 8
    trMaxEpoch =20

    # ---- Parameters related to image transforms: size of the down-scaled image, cropped image
    imgtransResize = 256
    imgtransCrop = 224

    pathModel = 'm-' + timestampLaunch + '.pth.tar'

    print('=== Training NN architecture = ', nnArchitecture, '===')
    ChexnetTrainer.train(pathImgTrain, pathImgVal, pathImgTest, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize,
                        trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, 10, 'BRACS_ROIRES', '/YourPath/max_f1.pth',None)

    #print('=== Testing the trained model ===')
    #print(pathModel)
    #ChexnetTrainer.test(pathImgTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize,
    #                     imgtransCrop, timestampLaunch)




# --------------------------------------------------------------------------------

if __name__ == '__main__':
 runTrain()  