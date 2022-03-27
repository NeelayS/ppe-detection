from backbone import Backbone
from model import CompleteModel


if __name__ == "__main__":

    model = CompleteModel(
        detection_threshold=0.5,
        detection_config="config/yolov3-tiny.cfg",
        detection_weights="weights/yolov3-tiny.weights",
        detection_reshape_size=(256, 128),
        classification_features_dim=16384,
        classification_layers_config=[1024, 512, 2],
        classification_n_heads=3,
    )
    outs = model("data/image_from_china(234).jpg")
    print(outs)
