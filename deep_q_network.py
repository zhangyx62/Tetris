import torch.nn as nn


class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        #     nn.ReLU(),
        # )
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        # )
        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
        #     nn.ReLU(),
        # )
        #
        # self.fc1 = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(7680, 40),
        # )

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(200, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 40),
        )
        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.fc1(x)

        x = self.net(x)

        return x
