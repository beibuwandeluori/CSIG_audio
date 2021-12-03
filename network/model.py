import torch
import torch.nn as nn
import torchvision
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import timm


def init_weights(module):
    """ Initialize the weights """
    if isinstance(module, (nn.Linear, nn.BatchNorm1d)):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=0.02)


def get_efficientnet(model_name='efficientnet-b0', num_classes=2):
    net = EfficientNet.from_pretrained(model_name)
    # net = EfficientNet.from_name(model_name)
    in_features = net._fc.in_features
    net._fc = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

    return net


def get_efficientnet_ns(model_name='tf_efficientnet_b0_ns', pretrained=True, num_classes=2):
    """
     # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    :param model_name:
    :param pretrained:
    :param num_classes:
    :return:
    """
    net = timm.create_model(model_name, pretrained=pretrained)
    n_features = net.classifier.in_features
    net.classifier = nn.Linear(n_features, num_classes)

    return net


def get_resnet(modelchoice='resnet50', num_classes=2):
    if modelchoice == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif modelchoice == 'resnet34':
        model = torchvision.models.resnet34(pretrained=True)
    elif modelchoice == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
    elif modelchoice == 'resnet101':
        model = torchvision.models.resnet101(pretrained=True)
    elif modelchoice == 'resnext101_32x8d':
        model = torchvision.models.resnext101_32x8d(pretrained=True)
    elif modelchoice == 'resnext50_32x4d':
        model = torchvision.models.resnext50_32x4d(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

    return model


def get_vgg(modelchoice='vgg16', num_classes=2):
    if modelchoice == 'vgg16':
        model = torchvision.models.vgg16(pretrained=True)
    elif modelchoice == 'vgg19':
        model = torchvision.models.vgg19(pretrained=True)
    # print(model)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

    return model


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, out_size, hidden_dim=512, dropout_prob=0.5, num_classes=82):
        super().__init__()
        self.norm = nn.BatchNorm1d(out_size)
        self.dense = nn.Linear(out_size, hidden_dim)
        self.norm_1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.dense_1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm_2 = nn.BatchNorm1d(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, num_classes)

    def forward(self, features):
        x = self.norm(features)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.relu(self.norm_1(x))
        x = self.dropout(x)
        x = self.dense_1(x)
        x = torch.relu(self.norm_2(x))
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class VGGish(nn.Module):
    """
    PyTorch implementation of the VGGish model.

    Adapted from: https://github.com/harritaylor/torch-vggish
    The following modifications were made: (i) correction for the missing ReLU layers, (ii) correction for the
    improperly formatted data when transitioning from NHWC --> NCHW in the fully-connected layers, and (iii)
    correction for flattening in the fully-connected layers.
    """

    def __init__(self):
        super(VGGish, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 24, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x).permute(0, 2, 3, 1).contiguous()
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class AudioModel(nn.Module):

    def __init__(self, vggish_path='/pubdata/chenby/py_project/CSIG_audio/network/pretrained_weights/pytorch_vggish.pth',
                 model_name='efficientnet-b0'):
        super(AudioModel, self).__init__()
        self.model_name = model_name

        self.feature_extractor = VGGish()
        if vggish_path is not None:
            self.feature_extractor.load_state_dict(torch.load(vggish_path, map_location='cpu'))
            print('VGGish found in {}'.format(vggish_path))

        if 'efficientnet' in self.model_name:
            self.efn = get_efficientnet(model_name=model_name)

    def forward(self, x):
        x = self.feature_extractor.features(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # (b, 512, 6, 4) -> (b, 6, 4, 512)
        # x = x.permute(0, 2, 1, 3).contiguous()  # (b, 512, 6, 4) -> (b, 6, 512, 4)  # hyj
        if 'efficientnet' in self.model_name:
            x = x.view(x.size(0), 3, 64, 64)
            x = self.efn(x)
        elif 'NextVLAD' in self.model_name:
            x = x.view(x.size(0), 24, 512)
            x = self.embed(x)
            # print(x.shape)
            x = self.fc(x)

        return x


class AudioModelV2(nn.Module):

    def __init__(self, vggish_path='/pubdata/chenby/py_project/CSIG_audio/network/pretrained_weights/pytorch_vggish.pth'):
        super(AudioModelV2, self).__init__()

        self.feature_extractor = VGGish()
        if vggish_path is not None:
            self.feature_extractor.load_state_dict(torch.load(vggish_path, map_location='cpu'))
            print('VGGish found in {}'.format(vggish_path))

        self.fc = ClassificationHead(out_size=128, num_classes=2)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc(x)

        return x


class AudioModelV3(nn.Module):

    def __init__(self, model_name='efficientnet-b0', num_classes=2):
        super(AudioModelV3, self).__init__()
        self.model_name = model_name

        if 'ns' in self.model_name:
            self.model_name = model_name
            self.model = get_efficientnet_ns(model_name=model_name, num_classes=num_classes)
            self.model.conv_stem.in_channels = 1
            self.model.conv_stem.weight = torch.nn.Parameter(self.model.conv_stem.weight[:, 0:1:, :, :])
        elif 'efficientnet' in model_name:
            self.model_name = model_name
            self.model = get_efficientnet(model_name=model_name, num_classes=num_classes)
            self.model._conv_stem.in_channels = 1
            self.model._conv_stem.weight = torch.nn.Parameter(self.model._conv_stem.weight[:, 0:1:, :, :])
        elif 'res' in self.model_name:
            self.model = get_resnet(modelchoice=model_name, num_classes=num_classes)
            self.model.conv1.in_channels = 1
            self.model.conv1.weight = torch.nn.Parameter(self.model.conv1.weight[:, 0:1:, :, :])
        elif 'vgg' in self.model_name:
            self.model = get_vgg(modelchoice=model_name, num_classes=num_classes)
            self.model.features[0].in_channels = 1
            self.model.features[0].weight = torch.nn.Parameter(self.model.features[0].weight[:, 0:1:, :, :])

        print(f'Model name is {self.model_name}!')

    def forward(self, x):
        x = self.model(x)

        return x


if __name__ == '__main__':
    # network = get_efficientnet('efficientnet-b0')
    # network = AudioModel(model_name='NextVLAD')
    # network = AudioModelV2()
    network = AudioModelV3(model_name='efficientnet-b2', num_classes=2)  # efficientnet-b0 resnet50
    network = network.to(torch.device('cpu'))
    print(network)

    from torchsummary import summary
    input_s = (1, 96, 64)
    print(summary(network, input_s, device='cpu'))
