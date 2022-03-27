from backbone import Backbone
from model import CompleteModel

import torch

if __name__ == "__main__":

    inp = torch.randn(1, 3, 256, 256)
    backbone = Backbone()
    out = backbone(inp)
    print(type(out))
