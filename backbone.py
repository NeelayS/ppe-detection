from torch import nn


class Backbone(nn.Module):
    def __init__(self, img_size=(256, 128)):
        super().__init__()

        self.backbone = nn.Identity()

    def forward(self, x):

        return self.backbone(x)


# Write backbone
