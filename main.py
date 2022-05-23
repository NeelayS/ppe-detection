import torch
import torch.nn.functional as F

from backbone import Backbone
from model import CompleteModel, Predictor


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str)
    args = parser.parse_args()

    model = CompleteModel(
        detection_threshold=0.05,
        detection_reshape_size=(96, 32),
        classification_features_dim=1536,
        classification_layers_config=[512, 128],
        classification_n_heads=3,
        classification_model_weights="../weights/best.pth",
    )
    outs = model(img_path=args.img)
    for out in outs:
        print(out)
        print(
            torch.argmax(F.softmax(out[0], dim=1), dim=1).item()
        )  # out indexed into 0 which corresponds to the classification head for hats. Similarly 1 for vest and 2 for boots
        print()

    predictor = Predictor(
        det_model_params={
            "threshold": 0.25,
        },
        class_model_params={
            "reshape_size": (96, 32),
            "features_dim": 1536,
            "layers_config": [512, 128],
            "n_heads": 3,
        },
        classification_model_weights="../weights/best.pth",
    )
    predictor(
        args.img, save_dir="results", task_id=1
    )  # task_id = 1 for vest classification
