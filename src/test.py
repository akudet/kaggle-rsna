from torchvision.datasets import *

if __name__ == "__main__":
    dataset = CocoDetection("../data/coco/train2014", "../data/coco/annotations/instances_train2014.json")
    print(dataset[0])
