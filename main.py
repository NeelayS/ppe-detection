import torch
import torch.nn.functional as F

from backbone import Backbone
from model import CompleteModel, Predictor


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str)
    args = parser.parse_args()

    # model = CompleteModel(
    #     detection_threshold=0.05,
    #     detection_reshape_size=(96, 32),
    #     classification_features_dim=1536,
    #     classification_layers_config=[512, 128],
    #     classification_n_heads=1,
    #     classification_model_weights="weights/best_model.pth",
    # )
    # outs = model(args.img)
    # for out in outs:
    #     print(out)
    #     print(torch.argmax(F.softmax(out[0], dim=1), dim=1).item())
    #     print()

    predictor = Predictor(
        det_model_params={
            "threshold": 0.25,
        },
        class_model_params={
            "reshape_size": (96, 32),
            "features_dim": 1536,
            "layers_config": [512, 128],
            "n_heads": 1,
        },
        classification_model_weights="../weights/best_model.pth",
    )
    predictor(args.img, save_dir="results")
