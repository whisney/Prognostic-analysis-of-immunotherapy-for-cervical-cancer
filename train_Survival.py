import os
from dataset_Survival import Dataset_ST
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import torch
from ResNet_3D import ResNet18_ST
import argparse
import shutil
from lifelines.utils import concordance_index
import nnet_survival_pytorch
from torch.optim.lr_scheduler import MultiStepLR

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='which gpu is used')
parser.add_argument('--bs', type=int, default=32, help='batch size')
parser.add_argument('--epoch', type=int, default=2000, help='all_epochs')
parser.add_argument('--task', type=str, default='OS', help='OS/PFS')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

lr_max = 0.0001
L2 = 0.0001
input_size = (16, 96, 96)
data_dir = 'Nii_Data_ROIs'
metadata_path = 'metadata/data_all.xlsx'
split_path = 'train_val_split.pkl'
breaks = np.arange(0, 41, 5)
n_intervals = len(breaks) - 1

save_dir = 'trained_models/Survival/{}_bs{}_epoch{}'.format(args.task, args.bs, args.epoch)

if os.path.exists(save_dir):
    shutil.rmtree(save_dir)

os.makedirs(save_dir, exist_ok=True)
train_writer = SummaryWriter(os.path.join(save_dir, 'log/train'), flush_secs=2)
val_writer = SummaryWriter(os.path.join(save_dir, 'log/val'), flush_secs=2)
print(save_dir)

print('dataset loading')

train_data = Dataset_ST(data_dir, split_path, metadata_path, breaks=breaks, data_set='train', task=args.task, augment=True)
val_data = Dataset_ST(data_dir, split_path, metadata_path, breaks=breaks, data_set='val', task=args.task, augment=False)

train_dataloader = DataLoader(dataset=train_data, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
val_dataloader = DataLoader(dataset=val_data, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

net = ResNet18_ST(in_channels=8, clinical_inchannels=26, n_classes=n_intervals, no_cuda=False).cuda()

loss_func = nnet_survival_pytorch.surv_likelihood(n_intervals)
optimizer = optim.Adam(net.parameters(), lr=lr_max, weight_decay=L2)
lr_scheduler = MultiStepLR(optimizer, milestones=[int((6 / 10) * args.epoch), int((9 / 10) * args.epoch)], gamma=0.1, last_epoch=-1)

best_Cindex_val = 0

print('training')

for epoch in range(args.epoch):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        break
    net.train()
    train_epoch_total_loss = []
    train_epoch_event_time = []
    train_epoch_event_indicator = []
    train_epoch_pred_scores = []
    for i, (imgs, modality_sign, Clinical_features, event_time, event_indicator, labels) in enumerate(train_dataloader):
        imgs, modality_sign, Clinical_features = imgs.cuda().float(), modality_sign.cuda().float(), Clinical_features.cuda().float()
        event_time, event_indicator, labels = event_time.cuda().float(), event_indicator.cuda().float(), labels.cuda().float()

        optimizer.zero_grad()
        out = net(imgs, modality_sign, Clinical_features)
        out = torch.sigmoid(out)
        loss = loss_func(y_pred=out, y_true=labels)
        loss.backward()
        optimizer.step()
        train_epoch_total_loss.append(loss.item())
        train_epoch_event_time.append(event_time.cpu())
        train_epoch_event_indicator.append(event_indicator.cpu())
        train_epoch_pred_scores.append(torch.sum(out, dim=1, keepdim=True).detach().cpu())
        print('[%d/%d, %d/%d] train_loss: %.3f' %
              (epoch + 1, args.epoch, i + 1, len(train_dataloader), loss.item()))
    lr_scheduler.step()

    with torch.no_grad():
        net.eval()
        val_epoch_total_loss = []
        val_epoch_event_time = []
        val_epoch_event_indicator = []
        val_epoch_pred_scores = []
        for i, (imgs, modality_sign, Clinical_features, event_time, event_indicator, labels) in enumerate(val_dataloader):
            imgs, modality_sign, Clinical_features = imgs.cuda().float(), modality_sign.cuda().float(), Clinical_features.cuda().float()
            event_time, event_indicator, labels = event_time.cuda().float(), event_indicator.cuda().float(), labels.cuda().float()
            out = net(imgs, modality_sign, Clinical_features)
            out = torch.sigmoid(out)
            loss = loss_func(y_pred=out, y_true=labels)
            val_epoch_total_loss.append(loss.item())
            val_epoch_event_time.append(event_time.cpu())
            val_epoch_event_indicator.append(event_indicator.cpu())
            val_epoch_pred_scores.append(torch.sum(out, dim=1, keepdim=True).detach().cpu())

    train_epoch_event_time = torch.cat(train_epoch_event_time, dim=0).numpy()[:, 0]
    train_epoch_event_indicator = torch.cat(train_epoch_event_indicator, dim=0).numpy()[:, 0]
    train_epoch_pred_scores = torch.cat(train_epoch_pred_scores, dim=0).numpy()[:, 0]

    val_epoch_event_time = torch.cat(val_epoch_event_time, dim=0).numpy()[:, 0]
    val_epoch_event_indicator = torch.cat(val_epoch_event_indicator, dim=0).numpy()[:, 0]
    val_epoch_pred_scores = torch.cat(val_epoch_pred_scores, dim=0).numpy()[:, 0]

    train_Cindex = concordance_index(event_times=train_epoch_event_time, predicted_scores=train_epoch_pred_scores, event_observed=train_epoch_event_indicator)
    val_Cindex = concordance_index(event_times=val_epoch_event_time, predicted_scores=val_epoch_pred_scores, event_observed=val_epoch_event_indicator)

    train_epoch_total_loss = np.mean(train_epoch_total_loss)
    val_epoch_total_loss = np.mean(val_epoch_total_loss)

    print(
        '[%d/%d] train_Cindex: %.3f val_Cindex: %.3f' %
        (epoch, args.epoch, train_Cindex, val_Cindex))

    if val_Cindex > best_Cindex_val:
        best_Cindex_val = val_Cindex
        torch.save(net.state_dict(), os.path.join(save_dir, 'best_Cindex_val.pth'))

    train_writer.add_scalar('lr', lr, epoch)
    train_writer.add_scalar('total_loss', train_epoch_total_loss, epoch)
    train_writer.add_scalar('Cindex', train_Cindex, epoch)

    val_writer.add_scalar('total_loss', val_epoch_total_loss, epoch)
    val_writer.add_scalar('Cindex', val_Cindex, epoch)

    if epoch + 1 == args.epoch:
        torch.save(net.state_dict(), os.path.join(save_dir, 'epoch' + str(epoch + 1) + '.pth'))

train_writer.close()
val_writer.close()
print('saved_model_name:', save_dir)