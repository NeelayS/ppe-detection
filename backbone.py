from torch import nn


class Backbone(nn.Module):
    def __init__(self, img_size=(256, 128)):
        super().__init__()

        self.backbone = nn.Identity()

    def forward(self, x):

        return self.backbone(x)


# Write backbone here if desired. By default, a pre-trained VGG model is used as the backbone in the model.py file
