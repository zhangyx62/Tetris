import torch.nn as nn


class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
        #     nn.ReLU(),
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.Flatten()
        # )
        #
        # self.fc1 = nn.Sequential(
        #     nn.Linear(960, 40),
        # )

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(200, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 40),
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
        # x = self.fc1(x)

        x = self.net(x)

        return x
