import argparse
import math
from sklearn.externals._packaging.version import PrePostDevType
from operator import add
import torch
import numpy as np

import glob
from data.dataset import DriveDataset
from torch.utils.data import DataLoader

from models.saunet import SA_UNet
from models.umix import U_Mixer
from utils.trainer import calculate_metrics
import segmentation_models_pytorch as smp


def inference(model, model_name, folder):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    predicted = []
    test_x = sorted(glob(f"{folder}/Images/*"))
    test_y = sorted(glob(f"{folder}/Labels/*"))
    test_dataset = DriveDataset(test_x, test_y, is_val=True)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
    for x, y in test_loader:
        with torch.no_grad():
            mask = torch.zeros(y.shape).to('cuda')
            y_pred = torch.zeros(y.shape).to('cuda')

            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            crop_count_h = int(math.ceil(y.shape[2] / 224))
            crop_count_w = int(math.ceil(y.shape[3] / 224))
            for i in range(crop_count_h):
                crop_h = i * 224
                for j in range(crop_count_w):
                    crop_w = j * 224
                    if crop_w + 224 > y.shape[3]:
                        crop_w = y.shape[3] - 224
                    if crop_h + 224 > y.shape[2]:
                        crop_h = y.shape[2] - 224

                    pred = model(x[:, :, crop_h: crop_h + 224, crop_w: crop_w + 224])
                    pred = torch.sigmoid(pred)
                    y_pred[:, :, crop_h: crop_h + 224, crop_w: crop_w + 224] += pred
                    mask[:, :, crop_h: crop_h + 224, crop_w: crop_w + 224] += torch.tensor(1)
        
            y_pred = y_pred / mask
            score = calculate_metrics(y, y_pred)
            metrics_score = list(map(add, metrics_score, score))
            predicted.append(y_pred)

    specificity = metrics_score[0]/len(test_loader)
    sensitivity = metrics_score[1]/len(test_loader)
    auc = metrics_score[2]/len(test_loader)
    f1 = metrics_score[3]/len(test_loader)
    acc = metrics_score[4]/len(test_loader)
    print(f'Model name: {model_name}')
    print(f'Specificity: {np.round(specificity, 4)}')
    print(f'Sensitivity: {np.round(sensitivity, 4)}')
    print(f'AUC ROC: {np.round(auc, 4)}')
    print(f'F1: {np.round(f1, 4)}')
    print(f'Accuracy: {np.round(acc, 4)}')

    return specificity, sensitivity, auc, f1, acc, predicted

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model", type=str, default="unet", help="name of model for downloading", choices=['unet', 'unetpp', 'saunet', 'umix']
    )
    parser.add_argument(
        "--ckpt_path", type=str, default='unet_drive.pth', help='path to checkpoint' 
    )
    parser.add_argument(
        "--model_name", type=str, default="unet_drive", help="model name for saving metrics"
    )
    parser.add_argument(
        "--image_folder", type=str, default="test_dataset", help="path to test dataset"
    )

    args = parser.parse_args()

    if args.model == 'unet':
        model = smp.Unet(
            encoder_name='resnet50',
            encoder_weights='imagenet',
            in_channels=3,
            classes=1
        )
        model.load_state_dict(torch.load(args.ckpt_path))

    elif args.model == 'unetpp':
        model = smp.UnetPlusPlus(
            encoder_name='resnet50',
            encoder_weights='imagenet',
            in_channels=3,
            classes=1
        )
        model.load_state_dict(torch.load(args.ckpt_path))

    elif args.model == 'saunet':
        model = SA_UNet()
        model.load_state_dict(torch.load(args.ckpt_path))

    elif args.model == 'umix':
        model = U_Mixer(img_size=224)
        model.load_state_dict(torch.load(args.ckpt_path))

    else:
        print("bad model")

    _ = inference(model, args.model_name, args.image_folder)