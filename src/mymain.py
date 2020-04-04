import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam

import dill
import argparse
from data import medicalDataLoader
from models.my_stacked_danet import DAF_stack
from models.basenet import Res_Deeplab
from common.progressBar import printProgressBar
from common.utils import (evaluate3D,
                          reconstruct3D,
                          saveImages_for3D,
                          getOneHotSegmentation,
                          predToSegmentation,
                          getTargetSegmentation,
                          computeDiceOneHot,
                          DicesToDice,
                          inference,
                          to_var
                          )
    
def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_normal(m.weight.data)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def runTraining(args):
    print('-' * 40)
    print('~~~~~~~~  Starting the training... ~~~~~~')
    print('-' * 40)

    batch_size = args.batch_size
    batch_size_val = 1
    batch_size_val_save = 1
    lr = args.lr
    base_lr=lr
    epoch = args.epochs
    root_dir = args.root
    model_dir = 'model'
    
    print(' Dataset: {} '.format(root_dir))

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set = medicalDataLoader.MedicalImageDataset('train',
                                                      root_dir,
                                                      transform=transform,
                                                      mask_transform=mask_transform,
                                                      augment=True,
                                                      equalize=False)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              num_workers=args.num_workers,
                              shuffle=True)

    val_set = medicalDataLoader.MedicalImageDataset('val',
                                                    root_dir,
                                                    transform=transform,
                                                    mask_transform=mask_transform,
                                                    equalize=False)

    val_loader = DataLoader(val_set,
                            batch_size=batch_size_val,
                            num_workers=args.num_workers,
                            shuffle=False)
                                                   
    val_loader_save_images = DataLoader(val_set,
                                        batch_size=batch_size_val_save,
                                        num_workers= args.num_workers,
                                        shuffle=False)

                                                                    
    # Initialize
    print("~~~~~~~~~~~ Creating the DAF Stacked model ~~~~~~~~~~")
    net = Res_Deeplab()
    print(" Model Name: {}".format('resDeeplab'))
    print(" Model ot create: resDeeplab")

    net.apply(weights_init)

    softMax = nn.Softmax()
    CE_loss = nn.CrossEntropyLoss()
    Dice_loss = computeDiceOneHot()
    mseLoss = nn.MSELoss()

    if torch.cuda.is_available():
        net.cuda()
        net = nn.DataParallel(net, device_ids=[0,1,2,3])
        softMax.cuda()
        CE_loss.cuda()
        Dice_loss.cuda()

    optimizer = Adam(net.parameters(), lr=lr, betas=(0.9, 0.99), amsgrad=False)

    BestDice, BestEpoch = 0, 0
    BestDice3D = [0,0,0,0]

    d1Val = []
    d2Val = []
    d3Val = []
    d4Val = []

    d1Val_3D = []
    d2Val_3D = []
    d3Val_3D = []
    d4Val_3D = []

    d1Val_3D_std = []
    d2Val_3D_std = []
    d3Val_3D_std = []
    d4Val_3D_std = []

    Losses = []

    print("~~~~~~~~~~~ Starting the training ~~~~~~~~~~")
    for i in range(epoch):
        net.train()
        lossVal = []

        totalImages = len(train_loader)
       
        for j, data in enumerate(train_loader):
            image, labels, img_names = data

            # prevent batchnorm error for batch of size 1
            if image.size(0) != batch_size:
                continue

            optimizer.zero_grad()
            MRI = to_var(image)
            Segmentation = to_var(labels)

            ################### Train ###################
            net.zero_grad()

            # Network outputs
            output0, output1 = net(MRI)

            segmentation_prediction = output0
            predClass_y = softMax(segmentation_prediction)

            Segmentation_planes = getOneHotSegmentation(Segmentation)

            segmentation_prediction_ones = predToSegmentation(predClass_y)

            # It needs the logits, not the softmax
            Segmentation_class = getTargetSegmentation(Segmentation)

            # Cross-entropy loss
            loss0 = CE_loss(output0, Segmentation_class)
            loss1 = CE_loss(output1, Segmentation_class)

            lossG = loss0+0.4*loss1

            # Compute the DSC
            DicesN, DicesB= Dice_loss(segmentation_prediction_ones, Segmentation_planes)

            DiceB = DicesToDice(DicesB)

            Dice_score = DiceB 

            lossG.backward()
            optimizer.step()
            
            lossVal.append(lossG.cpu().data.numpy())

            printProgressBar(j + 1, totalImages,
                             prefix="[Training] Epoch: {} ".format(i),
                             length=15,
                             suffix=" Mean Dice: {:.4f}, Dice1: {:.4f} , ".format(
                                 Dice_score.cpu().data.numpy(),
                                 DiceB.data.cpu().data.numpy(),))

      
        printProgressBar(totalImages, totalImages,
                             done="[Training] Epoch: {}, LossG: {:.4f}".format(i,np.mean(lossVal)))
       
        # Save statistics
        modelName = args.modelName
        directory = args.save_dir + modelName
        
        Losses.append(np.mean(lossVal))
        if i%5==0:
            d1 = inference(net, val_loader)

            d1Val.append(d1)

            if not os.path.exists(directory):
                os.makedirs(directory)

            np.save(os.path.join(directory, 'Losses.npy'), Losses)
            np.save(os.path.join(directory, 'd1Val.npy'), d1Val)

            currentDice = d1

            print("[val] DSC: (1): {:.4f} ".format(d1)) # MRI

            currentDice = currentDice.data.numpy()
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            torch.save(net.state_dict(), os.path.join(model_dir, "Latest_" + modelName + ".pth"),pickle_module=dill)

        # Evaluate on 3D
#        saveImages_for3D(net, val_loader_save_images, batch_size_val_save, 1000, modelName, False, False)
#        reconstruct3D(modelName, 1000, isBest=False)
#        DSC_3D = evaluate3D(modelName)

#        mean_DSC3D = np.mean(DSC_3D, 0)
#        std_DSC3D = np.std(DSC_3D,0)
#
#        d1Val_3D.append(mean_DSC3D[0])
#        d2Val_3D.append(mean_DSC3D[1])
#        d3Val_3D.append(mean_DSC3D[2])
#        d4Val_3D.append(mean_DSC3D[3])
#        d1Val_3D_std.append(std_DSC3D[0])
#        d2Val_3D_std.append(std_DSC3D[1])
#        d3Val_3D_std.append(std_DSC3D[2])
#        d4Val_3D_std.append(std_DSC3D[3])

#        np.save(os.path.join(directory, 'd0Val_3D.npy'), d1Val_3D)
#        np.save(os.path.join(directory, 'd1Val_3D.npy'), d2Val_3D)
#        np.save(os.path.join(directory, 'd2Val_3D.npy'), d3Val_3D)
#        np.save(os.path.join(directory, 'd3Val_3D.npy'), d4Val_3D)
#        
#        np.save(os.path.join(directory, 'd0Val_3D_std.npy'), d1Val_3D_std)
#        np.save(os.path.join(directory, 'd1Val_3D_std.npy'), d2Val_3D_std)
#        np.save(os.path.join(directory, 'd2Val_3D_std.npy'), d3Val_3D_std)
#        np.save(os.path.join(directory, 'd3Val_3D_std.npy'), d4Val_3D_std)


        if currentDice > BestDice:
            BestDice = currentDice

            BestEpoch = i
            
            if currentDice > 0.40:

 #               if np.mean(mean_DSC3D)>np.mean(BestDice3D):
#                    BestDice3D = mean_DSC3D

#                print("###    In 3D -----> MEAN: {}, Dice(1): {:.4f} Dice(2): {:.4f} Dice(3): {:.4f} Dice(4): {:.4f}   ###".format(np.mean(mean_DSC3D),mean_DSC3D[0], mean_DSC3D[1], mean_DSC3D[2], mean_DSC3D[3]))

                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Saving best model..... ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                torch.save(net.state_dict(), os.path.join(model_dir, "Best_" + modelName + ".pth"),pickle_module=dill)
 #               reconstruct3D(modelName, 1000, isBest=True)

        print("###                                                       ###")
        print("###    Best Dice: {:.4f} at epoch {} with Dice(1): {:.4f} Dice(2): {:.4f} Dice(3): {:.4f} Dice(4): {:.4f}   ###".format(BestDice, BestEpoch, d1,d2,d3,d4))
#        print("###    Best Dice in 3D: {:.4f} with Dice(1): {:.4f} Dice(2): {:.4f} Dice(3): {:.4f} Dice(4): {:.4f} ###".format(np.mean(BestDice3D),BestDice3D[0], BestDice3D[1], BestDice3D[2], BestDice3D[3] ))
        print("###                                                       ###")

        for param_group in optimizer.param_groups:
            lr = base_lr*((1-float(i)/epoch)**0.9)
            param_group['lr'] = lr
            print(' ----------  New learning Rate: {}'.format(lr))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--modelName',default = 'MS-Dual-Guided',type=str)
    parser.add_argument('--root', default = './CHAOS_MR/', type = str)
    parser.add_argument('--num_workers', default = 4, type = int)
    parser.add_argument('--save_dir', default = 'Results/Statistics/', type =str)
    parser.add_argument('--batch_size',default = 8,type = int)
    parser.add_argument('--epochs',default = 300,type = int)
    parser.add_argument('--lr',default = 0.001,type = float)
    args=parser.parse_args()
    runTraining(args)
