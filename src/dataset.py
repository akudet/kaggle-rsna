import pandas as pd
import torch.utils.data as data
import torchvision.datasets


class RSNADataset(data.Dataset):

    def __init__(self, root, anno_class_path, anno_bbox_path):
        self.anno_bbox = pd.read_csv(anno_bbox_path)
        self.anno_class = pd.read_csv(anno_class_path)

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass
