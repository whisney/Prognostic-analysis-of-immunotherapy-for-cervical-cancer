import torch
from torch import nn
from Networks import resnet

class ResNet18_ST(nn.Module):
    def __init__(self, in_channels, n_classes, clinical_inchannels, no_cuda=False):
        super(ResNet18_ST, self).__init__()
        self.backbone1 = resnet.resnet18(in_channels=in_channels, sample_input_W=1, sample_input_H=1, sample_input_D=1,
                                         shortcut_type='A', no_cuda=no_cuda, num_seg_classes=2)
        self.backbone2 = resnet.resnet18(in_channels=in_channels, sample_input_W=1, sample_input_H=1, sample_input_D=1,
                                         shortcut_type='A', no_cuda=no_cuda, num_seg_classes=2)
        self.backbone3 = resnet.resnet18(in_channels=in_channels, sample_input_W=1, sample_input_H=1, sample_input_D=1,
                                         shortcut_type='A', no_cuda=no_cuda, num_seg_classes=2)
        self.backbone4 = resnet.resnet18(in_channels=in_channels, sample_input_W=1, sample_input_H=1, sample_input_D=1,
                                         shortcut_type='A', no_cuda=no_cuda, num_seg_classes=2)

        self.avg_pool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.CF = nn.Sequential(
            nn.Linear(clinical_inchannels, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, 512, bias=True)
        )

        self.weight_linear = nn.Sequential(nn.Linear(2048, 2048, bias=True), nn.Sigmoid())
        self.fusion = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True))
        self.fusion_after = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True))

        self.out_linear = nn.Linear(512, n_classes)

    def feature_extraction(self, x, backbone):
        x = backbone.conv1(x)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)
        x = backbone.layer1(x)
        x = backbone.layer2(x)
        x = backbone.layer3(x)
        x = backbone.layer4(x)
        return x

    def forward(self, imgs, modality_sign, Clinical_features):
        x1 = self.feature_extraction(imgs[:, 0:8], self.backbone1)
        x2 = self.feature_extraction(imgs[:, 8:16], self.backbone2)
        x3 = self.feature_extraction(imgs[:, 16:24], self.backbone3)
        x4 = self.feature_extraction(imgs[:, 24:32], self.backbone4)

        x1 = self.avg_pool(x1)
        x2 = self.avg_pool(x2)
        x3 = self.avg_pool(x3)
        x4 = self.avg_pool(x4)

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x3 = x3.view(x3.size(0), -1)
        x4 = x4.view(x4.size(0), -1)

        x = torch.cat((x1.unsqueeze(1), x2.unsqueeze(1), x3.unsqueeze(1), x4.unsqueeze(1)), dim=1)
        x = x * modality_sign.unsqueeze(-1)

        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]

        x = torch.cat((x1, x2, x3, x4), dim=-1)

        w = self.weight_linear(x)
        w_x1, w_x2, w_x3, w_x4 = torch.split(w, 512, dim=-1)

        x = self.fusion(x)

        cf = self.CF(Clinical_features)

        x = x + cf + w_x1 * x1 + w_x2 * x2 + w_x3 * x3 + w_x4 * x4
        x = self.fusion_after(x)

        out = self.out_linear(x)
        return out