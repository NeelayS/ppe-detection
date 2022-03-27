from torch import nn
import torch.nn.functional as F

from backbone import Backbone


class ClassificationHead(nn.Module):
    def __init__(self, features_size, hidden_dim):
        super().__init__()

        self.flattened_dim = features_size.shape[0] * features_size.shape[1]

        self.head = nn.Sequential(
            nn.Linear(self.flattened_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):

        x = x.view(self.flattened_dim, -1)

        return self.head(x)


class DetectionModel:
    pass


class ClassificationModel(nn.Module):
    def __init__(self, reshape_size, features_size, hidden_dim, n_heads):
        super().__init__()

        self.reshape_size = reshape_size

        self.backbone = Backbone(img_size=reshape_size)

        self.heads = nn.ModuleList()
        for n in range(n_heads):
            self.heads.append(
                ClassificationHead(img_size=features_size, hidden_dim=hidden_dim)
            )

        def forward(self, detections):

            outs = []

            for detection in detections:

                reshaped_detection = F.interpolate(
                    detection, self.reshape_size, align_corners=True
                )
                features = self.backbone(reshaped_detection)

                detection_outs = []
                for head in self.heads:
                    out = head(features)
                    detection_outs.append(out)

                outs.append(detection_outs)

            return outs


class CompleteModel(nn.Module):
    def __init__(
        self, detection_threshold, reshape_size, features_size, hidden_dim, n_heads
    ):
        super().__init__()

        self.detection_model = DetectionModel(detection_threshold=detection_threshold)
        self.classification_model = ClassificationModel(
            reshape_size=reshape_size,
            features_size=features_size,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
        )

    def forward(self, img):

        # img = preprocess_img(img)
        detections = self.detection_model(img)
        # filtered_detections = filter_detections(detections)
        # processed_detections = process_detections(filtered_detections)

        img_outs = self.classification_model(detections)

        return img_outs


# Map outs for a detection uniquely to the detection
# Threshold, filter, and process detections
# Pre-process images
# Detection model
# Load pre-trained weights
