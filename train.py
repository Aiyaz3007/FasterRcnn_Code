from pycocotools.coco import COCO
from torchvision.datasets import CocoDetection
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import json
import torch

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
import warnings
warnings.filterwarnings("ignore")

from utils import train_epoch,CustomDataset,plot_graph


batch_size = 40


trainannotationFile = "/data1/Desktop/Interns/aiyaz/fasterRcnn/mask_dataset/Mask_Detection_Coco_dataset/train.json"
valannotationFile = "/data1/Desktop/Interns/aiyaz/fasterRcnn/mask_dataset/Mask_Detection_Coco_dataset/val.json"

root_dir = "/data1/Desktop/Interns/aiyaz/fasterRcnn/mask_dataset/Mask_Detection_Coco_dataset/images"


def collate_fn(batch):
    return tuple(zip(*batch))



train_dataset = CustomDataset(annotationFile=trainannotationFile, root_dir=root_dir)
val_dataset = CustomDataset(annotationFile=valannotationFile, root_dir=root_dir)

train_data_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=collate_fn)
val_data_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=collate_fn)




classes = [
    "__background__",
    "Mask_Weared",
    "Mask_Weared_Incorrect",
    "Mask_Not_Weared"
]


# model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(classes))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001)

print("Device using :",device)

losses = train_epoch(train_dataloader=train_data_loader,
                    val_dataloader=val_data_loader,
                    epochs=150,
                    model=model,
                    optimizer=optimizer,
                    device=device,
                    save_model=True)


# with open("losses.txt","w") as f:
#    f.write(str(losses[0]))
#    f.write(str(losses[1]))
# plot_graph(values=losses)