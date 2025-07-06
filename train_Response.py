import os
from dataset_Response import Dataset_ST
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import torch
from ResNet_3D import ResNet18_ST
import argparse
import shutil
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import MultiStepLR

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='which gpu is used')
parser.add_argument('--bs', type=int, default=32, help='batch size')
parser.add_argument('--epoch', type=int, default=200, help='all_epochs')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

lr_max = 0.00001
L2 = 0.0001
input_size = (16, 96, 96)
data_dir = 'Nii_Data_ROIs'
metadata_path = 'metadata/Response.xlsx'
split_path = 'train_val_split.pkl'

save_dir = 'trained_models/Response/bs{}_epoch{}'.format(args.bs, args.epoch)

if os.path.exists(save_dir):
    shutil.rmtree(save_dir)

os.makedirs(save_dir, exist_ok=True)
train_writer = SummaryWriter(os.path.join(save_dir, 'log/train'), flush_secs=2)
val_writer = SummaryWriter(os.path.join(save_dir, 'log/val'), flush_secs=2)
print(save_dir)

print('dataset loading')

train_data = Dataset_ST(data_dir, split_path, metadata_path, data_set='train', augment=True)
val_data = Dataset_ST(data_dir, split_path, metadata_path, data_set='val', augment=False)

train_dataloader = DataLoader(dataset=train_data, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
val_dataloader = DataLoader(dataset=val_data, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

net = ResNet18_ST(in_channels=8, clinical_inchannels=26, n_classes=2, no_cuda=False).cuda()

loss_func = nn.CrossEntropyLoss()
optimizer = optim.AdamW(net.parameters(), lr=lr_max, weight_decay=L2)
lr_scheduler = MultiStepLR(optimizer, milestones=[int((6 / 10) * args.epoch), int((9 / 10) * args.epoch)], gamma=0.1, last_epoch=-1)

best_AUC_val = 0

print('training')

for epoch in range(args.epoch):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        break
    net.train()
    train_epoch_total_loss = []
    train_epoch_one_hot_label = []
    train_epoch_pred_scores = []
    for i, (imgs, modality_sign, Clinical_features, label) in enumerate(train_dataloader):
        imgs, modality_sign, Clinical_features = imgs.cuda().float(), modality_sign.cuda().float(), Clinical_features.cuda().float()
        label = label.cuda().long()
        labels_one_hot = torch.zeros((label.size(0), 2)).cuda().scatter_(1, label.unsqueeze(1), 1).float().cpu()
        optimizer.zero_grad()
        out = net(imgs, modality_sign, Clinical_features)
        loss = loss_func(out, label)
        loss.backward()
        optimizer.step()
        out = torch.softmax(out, dim=1)
        train_epoch_total_loss.append(loss.item())
        train_epoch_one_hot_label.append(labels_one_hot)
        train_epoch_pred_scores.append(out.detach().cpu())
        print('[%d/%d, %d/%d] train_loss: %.3f' %
              (epoch + 1, args.epoch, i + 1, len(train_dataloader), loss.item()))
    lr_scheduler.step()

    with torch.no_grad():
        net.eval()
        val_epoch_total_loss = []
        val_epoch_one_hot_label = []
        val_epoch_pred_scores = []
        for i, (imgs, modality_sign, Clinical_features, label) in enumerate(val_dataloader):
            imgs, modality_sign, Clinical_features = imgs.cuda().float(), modality_sign.cuda().float(), Clinical_features.cuda().float()
            label = label.cuda().long()
            labels_one_hot = torch.zeros((label.size(0), 2)).cuda().scatter_(1, label.unsqueeze(1), 1).float().cpu()
            out = net(imgs, modality_sign, Clinical_features)
            loss = loss_func(out, label)
            out = torch.softmax(out, dim=1)
            val_epoch_total_loss.append(loss.item())
            val_epoch_one_hot_label.append(labels_one_hot)
            val_epoch_pred_scores.append(out.detach().cpu())

    train_epoch_one_hot_label = torch.cat(train_epoch_one_hot_label, dim=0).numpy().astype(np.uint8)
    train_epoch_pred_scores = torch.cat(train_epoch_pred_scores, dim=0).numpy()

    val_epoch_one_hot_label = torch.cat(val_epoch_one_hot_label, dim=0).numpy().astype(np.uint8)
    val_epoch_pred_scores = torch.cat(val_epoch_pred_scores, dim=0).numpy()

    train_AUC = roc_auc_score(train_epoch_one_hot_label, train_epoch_pred_scores)
    val_AUC = roc_auc_score(val_epoch_one_hot_label, val_epoch_pred_scores)

    train_epoch_total_loss = np.mean(train_epoch_total_loss)
    val_epoch_total_loss = np.mean(val_epoch_total_loss)

    print(
        '[%d/%d] train_AUC: %.3f val_AUC: %.3f' %
        (epoch, args.epoch, train_AUC, val_AUC))

    if val_AUC > best_AUC_val:
        best_AUC_val = val_AUC
        torch.save(net.state_dict(), os.path.join(save_dir, 'best_AUC_test.pth'))

    train_writer.add_scalar('lr', lr, epoch)
    train_writer.add_scalar('total_loss', train_epoch_total_loss, epoch)
    train_writer.add_scalar('AUC', train_AUC, epoch)

    val_writer.add_scalar('total_loss', val_epoch_total_loss, epoch)
    val_writer.add_scalar('AUC', val_AUC, epoch)

    if epoch + 1 == args.epoch:
        torch.save(net.state_dict(), os.path.join(save_dir, 'epoch' + str(epoch + 1) + '.pth'))

train_writer.close()
val_writer.close()
print('saved_model_name:', save_dir)