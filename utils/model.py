import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

class ClassC(nn.Module):
    def __init__(self, num_classes=10, nc=3, ch=64, im_size=32):
        super(ClassC, self).__init__()
        
        self.conv1 = nn.Conv2d(nc, ch, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(ch, ch, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(ch, ch, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.fc = nn.Linear(int(ch * im_size*im_size/(16*16)), num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, latent_flag = False):

        out = self.conv1(x)

        out = self.conv2(F.max_pool2d(F.relu(out), 2))

        out = self.conv3(F.max_pool2d(F.relu(out), 2))

        out = self.conv4(F.max_pool2d(F.relu(out), 2))

        out = F.max_pool2d(F.relu(out), 2)

        feature = out.view(out.size(0), -1)
        
        score = self.fc(feature)
        
        if latent_flag:
            return score, feature            
        else:
            return score

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, model_root=None, **kwargs):
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet'], model_root), strict=False)
    return model