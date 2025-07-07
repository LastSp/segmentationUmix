import argparse
import glob
import time
import numpy as np
import torch

from data.dataset import DriveDataset
from torch.utils.data import DataLoader

from models.saunet import SA_UNet
from models.umix import U_Mixer
from utils.losses import BCEDiceLoss
from utils.trainer import evaluate, score, train
import segmentation_models_pytorch as smp


def train_model(model, save_dir, model_name, encoder_name, data_dir, lr, metrics, wd=0.0, epochs=100, epoch_mod=50, batch_size=1, with_aug=False):
    checkpoint_path = f'{save_dir}/{model_name}_{encoder_name}_val.pth'

    name = model_name + '_' + encoder_name

    train_image_path = f'{data_dir}/Images'
    train_labels_path = f'{data_dir}/Labels'

    all_images = np.array(sorted(glob(f"{train_image_path}/*")))
    all_labels = np.array(sorted(glob(f"{train_labels_path}/*")))

    order = np.random.permutation(len(all_images))

    num_train = int(len(all_images)*0.8)

    train_x = all_images[order[:num_train]]
    train_y = all_labels[order[:num_train]]

    valid_x = all_images[order[num_train:]]
    valid_y = all_labels[order[num_train:]]

    train_dataset = DriveDataset(train_x, train_y, is_val=False)
    valid_dataset = DriveDataset(valid_x, valid_y, is_val=True)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss = BCEDiceLoss()

    best_loss = float('inf')
    tr_losses = []
    val_losses = []
    for epoch in range(1, 1+epochs):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, loss, device)
        valid_loss = evaluate(model, valid_loader, loss, device)

        tr_losses.append(train_loss)
        val_losses.append(valid_loss)

        if valid_loss < best_loss:
            print(f'New best model. Loss improved from {best_loss} to {valid_loss}')
            print()
            best_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

        end_time = time.time()
        print(f'Epoch {epoch} | Epoch time: {np.round(end_time - start_time, 2)}')
        print(f'Train loss: {train_loss}')
        print(f'Val loss: {valid_loss}')
        print()

        if epoch % 5 == 0:
            metrics_train = score(model, train_loader, device)
            metrics['spec'][name]['train'][epoch] = metrics_train[0]
            metrics['sens'][name]['train'][epoch] = metrics_train[1]
            metrics['auc'][name]['train'][epoch] = metrics_train[2]
            metrics['f1'][name]['train'][epoch] = metrics_train[3]
            metrics['acc'][name]['train'][epoch] = metrics_train[4]

            metrics_ = score(model, valid_loader, device)
            metrics['spec'][name]['valid'][epoch] = metrics_[0]
            metrics['sens'][name]['valid'][epoch] = metrics_[1]
            metrics['auc'][name]['valid'][epoch] = metrics_[2]
            metrics['f1'][name]['valid'][epoch] = metrics_[3]
            metrics['acc'][name]['valid'][epoch] = metrics_[4]


        if epoch >= 50 and epoch % epoch_mod == 0:
            check_path = f'{save_dir}' + '/' + f'{model_name}_{encoder_name}_val_{epoch}.pth'
            torch.save(model.state_dict(), check_path)
            print()

    return tr_losses, val_losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model", type=str, default="unet", help="name of model for downloading", choices=['unet', 'unetpp', 'saunet', 'umix']
    )
    parser.add_argument(
        "--ckpt_path", type=str, default='', help='path to checkpoint' 
    )
    parser.add_argument(
        "--model_name", type=str, default="unet_drive", help="model name for saving metrics"
    )
    parser.add_argument(
        "--image_folder", type=str, default="test_dataset", help="path to test dataset"
    )
    parser.add_argument(
        "--save_dir", type=str, default="my_results"
    )
    parser.add_argument(
        "--lr", type=int, default=0.0002
    )
    parser.add_argument(
        "--epochs", type=int, default=100
    )
    parser.add_argument(
        "--batch_size", type=int, default=1
    )

    args = parser.parse_args()
    
    if args.model == 'unet':
        model = smp.Unet(
            encoder_name='resnet50',
            encoder_weights='imagenet',
            in_channels=3,
            classes=1
        )

    elif args.model == 'unetpp':
        model = smp.UnetPlusPlus(
            encoder_name='resnet50',
            encoder_weights='imagenet',
            in_channels=3,
            classes=1
        )

    elif args.model == 'saunet':
        model = SA_UNet()

    elif args.model == 'umix':
        model = U_Mixer(img_size=224)

    if args.ckpt_path:
        model.load_state_dict(torch.load(args.ckpt_path))

    metrics = {'spec': {},
           'sens': {},
           'auc': {},
           'f1': {},
           'acc': {},
           'psnr': {}}

    metrics['spec'][args.model_name] = {'train': {}, 'valid': {}}
    metrics['sens'][args.model_name] = {'train': {}, 'valid': {}}
    metrics['auc'][args.model_name] = {'train': {}, 'valid': {}}
    metrics['f1'][args.model_name] = {'train': {}, 'valid': {}}
    metrics['acc'][args.model_name] = {'train': {}, 'valid': {}}
    metrics['psnr'][args.model_name] = {'train': {}, 'valid': {}}

    tr_loss, val_loss = train_model(model, args.save_dir, args.model_name, '_train', args.image_folder, args.lr, )
    
