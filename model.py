import cv2
import os
from pytorchyolo import detect, models as yolo_models
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import io, models

from backbone import Backbone


class ClassificationHead(nn.Module):
    def __init__(self, features_dim, layers_config=[1024, 512]):
        super().__init__()

        self.features_dim = features_dim

        layers_config = [features_dim] + list(layers_config)

        self.head = nn.ModuleList()
        for i in range(0, len(layers_config) - 1):
            self.head.append(
                nn.Sequential(
                    nn.Linear(layers_config[i], layers_config[i + 1]),
                    nn.ReLU(),
                ),
            )
        self.head.append(nn.Linear(layers_config[-1], 2))
        self.head = nn.Sequential(*self.head)

    def forward(self, x):

        x = torch.flatten(x, 1)

        return self.head(x)


class YoloV3DetectionModel:
    def __init__(self, config_path, weights_path, threshold=0.5):

        self.threshold = threshold

        self.model = yolo_models.load_model(config_path, weights_path)

    def _filter_detections(self, detections):

        filtered_detections = []

        for detection in detections:

            if (
                detection[-2] > self.threshold and int(detection[-1]) == 0
            ):  # 0 = person class
                filtered_detection = list(map(lambda x: max(int(x), 0), detection[:-2]))
                filtered_detections.append(filtered_detection)

        return filtered_detections

    def __call__(self, img_path, return_detections=False):

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        detections = detect.detect_image(self.model, img)
        detections = self._filter_detections(detections)

        print(f"Detected {len(detections)} persons")

        img = io.read_image(img_path)

        cropped_detections = []
        for detection in detections:
            cropped_detections.append(
                img[:, detection[1] : detection[3], detection[0] : detection[2]]
                .unsqueeze(0)
                .float()
            )

        if return_detections:
            return cropped_detections, detections

        return cropped_detections


class ClassificationModel(nn.Module):
    def __init__(
        self, reshape_size, features_dim, layers_config, n_heads, backbone=None
    ):
        super().__init__()

        self.reshape_size = reshape_size

        if backbone is None:
            self.backbone = models.vgg19(pretrained=False).features
        else:
            self.backbone = backbone

        self.heads = nn.ModuleList()
        for _ in range(n_heads):
            self.heads.append(
                ClassificationHead(
                    features_dim=features_dim, layers_config=layers_config
                )
            )

    def forward(self, detections):

        outs = []

        for detection in detections:

            detection = F.interpolate(
                detection, self.reshape_size, mode="bilinear", align_corners=True
            )
            features = self.backbone(detection)

            instance_outs = []

            for head in self.heads:
                out = head(features)
                instance_outs.append(out)

            outs.append(instance_outs)

        return outs


class CompleteModel(nn.Module):
    def __init__(
        self,
        detection_threshold,
        detection_config,
        detection_weights,
        detection_reshape_size,
        classification_features_dim,
        classification_layers_config,
        classification_n_heads,
        classification_model_weights=None,
    ):
        super().__init__()

        self.detection_model = YoloV3DetectionModel(
            config_path=detection_config,
            weights_path=detection_weights,
            threshold=detection_threshold,
        )
        self.classification_model = ClassificationModel(
            reshape_size=detection_reshape_size,
            features_dim=classification_features_dim,
            layers_config=classification_layers_config,
            n_heads=classification_n_heads,
        )

        if classification_model_weights is not None:
            self.classification_model.load_state_dict(
                torch.load(classification_model_weights, map_location="cpu")
            )

    def forward(self, img_path):

        detections = self.detection_model(img_path)
        img_outs = self.classification_model(detections)

        return img_outs


class Predictor:
    def __init__(
        self, det_model_params, class_model_params, classification_model_weights=None
    ):

        self.det_model = YoloV3DetectionModel(**det_model_params)
        self.class_model = ClassificationModel(**class_model_params)

        if classification_model_weights is not None:
            self.class_model.load_state_dict(
                torch.load(classification_model_weights, map_location="cpu")
            )

        self.class_model.eval()

    def __call__(self, img_path, task_id=0, save_dir="."):

        cropped_detections, detections = self.det_model(
            img_path, return_detections=True
        )

        with torch.no_grad():

            img_outs = self.class_model(cropped_detections)

            classifications = []
            for img_out in img_outs:
                classifications.append(
                    torch.argmax(F.softmax(img_out[task_id], dim=1), dim=1).item()
                )

        img = cv2.imread(img_path)
        for detection, classification in zip(detections, classifications):

            if classification == 1:
                cv2.rectangle(
                    img,
                    (detection[0], detection[1]),
                    (detection[2], detection[3]),
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.rectangle(
                    img,
                    (detection[0], detection[1]),
                    (detection[2], detection[3]),
                    (0, 0, 255),
                    2,
                )

        save_path = os.path.join(save_dir, "output_" + os.path.basename(img_path))
        cv2.imwrite(save_path, img)


# Map outs for a detection uniquely to the detection
# Pre-process images
# Detection model
# Load pre-trained weights
