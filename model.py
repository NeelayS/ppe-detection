import cv2
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import io, models

from backbone import Backbone


class PoseDetectionModel:
    def __init__(self, device=torch.device("cpu"), threshold=0.5):

        self.model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).eval()
        self.model.to(device)

        self.device = device
        self.threshold = threshold

    def _filter_detections(self, results):

        filtered_detections = []

        for i in range(len(results)):

            if results["scores"][i] < self.threshold or int(results["labels"][i]) != 1:
                continue

            keypoints = results["keypoints"][i]

            if (
                int(keypoints[0][2])
                == int(keypoints[-2][2])
                == int(keypoints[-1][2])
                == 1
            ):

                det_box = [max(0, int(coord)) for coord in results["boxes"][i]]
                filtered_detections.append(det_box)

        return filtered_detections

    def __call__(self, img=None, img_path=None, return_detections=False):

        assert (
            img is not None or img_path is not None
        ), "Either img or img_path must be provided"

        if img is None:
            img = io.read_image(img_path)

        if not isinstance(img, torch.Tensor):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).permute(2, 0, 1).float()

        if len(img.shape) == 3:
            img = img.unsqueeze(0)

        n_img = img / 255.0
        n_img = n_img.to(self.device)

        out = self.model(n_img)[0]
        filtered_detections = self._filter_detections(out)

        img = img.squeeze(0)
        cropped_detections = []

        for detection in filtered_detections:

            cropped_detections.append(
                img[:, detection[1] : detection[3], detection[0] : detection[2]]
                .unsqueeze(0)
                .float()
            )

        if return_detections:
            return cropped_detections, filtered_detections

        return cropped_detections


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


# class YoloV3DetectionModel:
#     def __init__(self, config_path, weights_path, threshold=0.5):

#         self.threshold = threshold

#         self.model = yolo_models.load_model(config_path, weights_path)

#     def _filter_detections(self, detections):

#         filtered_detections = []

#         for detection in detections:

#             if (
#                 detection[-2] > self.threshold and int(detection[-1]) == 0
#             ):  # 0 = person class
#                 filtered_detection = list(map(lambda x: max(int(x), 0), detection[:-2]))
#                 filtered_detections.append(filtered_detection)

#         return filtered_detections

#     def __call__(self, img_path, return_detections=False):

#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         detections = detect.detect_image(self.model, img)
#         detections = self._filter_detections(detections)

#         print(f"Detected {len(detections)} persons")

#         img = io.read_image(img_path)

#         cropped_detections = []
#         for detection in detections:
#             cropped_detections.append(
#                 img[:, detection[1] : detection[3], detection[0] : detection[2]]
#                 .unsqueeze(0)
#                 .float()
#             )

#         if return_detections:
#             return cropped_detections, detections

#         return cropped_detections


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
        detection_reshape_size,
        classification_features_dim,
        classification_layers_config,
        classification_n_heads,
        classification_model_weights=None,
        device=torch.device("cpu"),
    ):
        super().__init__()

        self.detection_model = PoseDetectionModel(
            threshold=detection_threshold, device=device
        )
        self.classification_model = ClassificationModel(
            reshape_size=detection_reshape_size,
            features_dim=classification_features_dim,
            layers_config=classification_layers_config,
            n_heads=classification_n_heads,
        ).to(device)

        if classification_model_weights is not None:
            self.classification_model.load_state_dict(
                torch.load(classification_model_weights, map_location="cpu")
            )

        self.device = device

    def forward(self, img=None, img_path=None):

        assert (
            img is not None or img_path is not None
        ), "Either img or img_path must be provided"

        if img is None:
            img = io.read_image(img_path)

        if not isinstance(img, torch.Tensor):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).permute(2, 0, 1).float()

        if len(img.shape) == 3:
            img = img.unsqueeze(0)

        img = img.to(self.device)

        cropped_detections = self.detection_model(img=img, img_path=img_path)
        img_outs = self.classification_model(cropped_detections)

        return img_outs


class Predictor:
    def __init__(
        self,
        det_model_params,
        class_model_params,
        classification_model_weights=None,
        device=torch.device("cpu"),
    ):

        self.det_model = PoseDetectionModel(**det_model_params)
        self.class_model = ClassificationModel(**class_model_params).to(device)

        if classification_model_weights is not None:
            self.class_model.load_state_dict(
                torch.load(classification_model_weights, map_location="cpu")
            )

        self.class_model.eval()

    def __call__(self, img_path, task_id=0, save_dir="."):

        cropped_detections, detections = self.det_model(
            img_path=img_path, return_detections=True
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
