import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import warnings
warnings.filterwarnings("ignore")
import os
from random import choice
import json

classes = [
 "__background__",
 "Mask_Weared",
 "Mask_Weared_Incorrect",
 "Mask_Not_Weared"
]

# model_file_path = "/data1/Desktop/Interns/aiyaz/fasterRcnn/models/model_28_1.95.pth"
# model_file_path = "/data1/Desktop/Interns/aiyaz/fasterRcnn/models/model_32_1.96.pth"
model_file_path = "/data1/Desktop/Interns/aiyaz/fasterRcnn/models/model_14_0.38.pth"
model_pred = fasterrcnn_resnet50_fpn(pretrained=False)
in_features = model_pred.roi_heads.box_predictor.cls_score.in_features
model_pred.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(classes))
device = torch.device('cpu')
model_pred.to(device)

model_pred.load_state_dict(torch.load(model_file_path, map_location=device))
model_pred.eval() 

from PIL import Image
# Define the transformation to preprocess the input image
transform = T.Compose([T.ToTensor()])

# Load and preprocess the input image
# image_file_path = "/data1/Desktop/Interns/aiyaz/fasterRcnn/mask_dataset/Mask_Detection_Coco_dataset/images/maksssksksss114.png"
# image_file_path = "/data1/Desktop/Interns/aiyaz/fasterRcnn/mask_dataset/Mask_Detection_Coco_dataset/images/maksssksksss123.png"
# image_file_path = "/data1/Desktop/Interns/aiyaz/fasterRcnn/mask_dataset/Mask_Detection_Coco_dataset/images/maksssksksss135.png"
with open("/data1/Desktop/Interns/aiyaz/fasterRcnn/mask_dataset/Mask_Detection_Coco_dataset/val.json","r") as f:
    data = json.load(f)
images = [x["file_name"] for x in data["images"]]
image_file_path = os.path.join("/data1/Desktop/Interns/aiyaz/fasterRcnn/mask_dataset/Mask_Detection_Coco_dataset/images",choice(images))

image = cv2.imread(image_file_path)
# image = Image.open(image_file_path)
input_image = transform(image)

# Add batch dimension since Faster R-CNN expects a batch of images
input_image = input_image.unsqueeze(0)

# Move the input tensor to the same device as the model (CPU in this case)
input_image = input_image.to(device)

with torch.no_grad():
    predictions = model_pred(input_image)

import numpy as np
import cv2

image_np = np.array(image)

# Draw bounding boxes on the image
for box, label, score in zip(predictions[0]["boxes"], predictions[0]["labels"], predictions[0]["scores"]):
    if score >=0.5 : # Only draw boxes with confidence greater than 0.5 (you can adjust this threshold)
        print(label)
        box = box.to(torch.int64).cpu().numpy()
        label = label.item()
        print(score)
        color = (0, 255, 255) if label == 1 else (0,0,255) if label == 2 else (255,0,0) # Green color for the bounding boxes
        thickness = 1 # Thickness of the bounding box lines
        # print(f"{score:.2f}")
        cv2.rectangle(image_np, (box[0], box[1]), (box[2], box[3]), color, thickness)
        cv2.putText(image_np, f"{classes[label]} {score:.2f}", (box[0], box[1] - 10),
                        cv2.FONT_ITALIC, 0.5, color, thickness)


# cv2.imshow("frame",image_np)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite(filename="output.jpg",img=image_np)