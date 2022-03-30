from torch.utils.data import Dataset
from torchvision import io
from os.path import join


class PictorPPEDataset(Dataset):
    def __init__(self, img_dir, annotations_list):
        super().__init__()

        self.img_dir = img_dir

        with open(annotations_list, "rb") as f:
            self.annotations = sorted(f.readlines())
        f.close()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):

        annotation = str(self.annotations[idx])[2:-5].split("\\t")

        img_name = annotation[0]
        img = io.read_image(join(self.img_dir, img_name))

        cropped_detections = []
        labels = []

        for det_annotation in annotation[1:]:

            label = int(det_annotation[-1])
            labels.append(label)

            x1, y1, x2, y2 = list(map(lambda x: int(x), det_annotation[:-2].split(",")))
            cropped_detection = img[:, y1:y2, x1:x2]
            cropped_detections.append(cropped_detection)

        return cropped_detections, labels
