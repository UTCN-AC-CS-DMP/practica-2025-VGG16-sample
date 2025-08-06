import torch.nn as nn


class VGG16(nn.Module):
    def __init__(self, num_classes=100):
        super(VGG16, self).__init__()

        def block(in_channels, out_channels, pool=False):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            block(3, 64),
            block(64, 64, pool=True),
            block(64, 128),
            block(128, 128, pool=True),
            block(128, 256),
            block(256, 256),
            block(256, 256, pool=True),
            block(256, 512),
            block(512, 512),
            block(512, 512, pool=True),
            block(512, 512),
            block(512, 512),
            block(512, 512, pool=True),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
