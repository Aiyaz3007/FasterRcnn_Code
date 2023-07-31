# custom training function
from tqdm import tqdm
import torch
import numpy as np
import cv2
from pycocotools.coco import COCO
import json
import os
import matplotlib.pyplot as plt
import os


def train_epoch(train_dataloader,
                val_dataloader,
                epochs:int,
                model,
                optimizer,
                device,
                save_model=False):
    train_loss_arr = []
    val_loss_arr = []
    def one_epoch_train(dataloader,
                        epoch:int,
                        model,
                        optimizer,
                        device):
        total_loss=[]
        total_loss_dict=[]
        one_epoch_bar = tqdm(total=len(dataloader),desc=f"epoch {epoch}",leave=True)
        for batch,(images,targets) in enumerate(dataloader):
            images = [image.to(device) for image in images]
            targets = [{k:torch.tensor(v).to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images,targets)
            losses = sum([loss for loss in loss_dict.values()])
            losses_dict = [{k:v.item()} for k,v in loss_dict.items()]

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss.append(losses.item())
            # total_loss_dict.append(losses_dict)
            one_epoch_bar.update(1)
        return np.mean(total_loss)
 
    def one_epoch_val(dataloader,
                    model,
                    device):
        with torch.no_grad():
            total_loss=[]
            total_loss_dict=[]
            one_epoch_bar = tqdm(total=len(dataloader),desc=f"validation",leave=True)

        for batch,(images,targets) in enumerate(dataloader):
            images = [image.to(device) for image in images]
            targets = [{k:torch.tensor(v).to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images,targets)
            losses = sum([loss for loss in loss_dict.values()])
            total_loss.append(losses.item())
            one_epoch_bar.update(1)

        return np.mean(total_loss)
  
    for epoch in range(epochs):
        train_loss = one_epoch_train(dataloader=train_dataloader,
                                    epoch=epoch+1,
                                    model=model,
                                    optimizer=optimizer,
                                    device=device)
        val_loss = one_epoch_val(dataloader=val_dataloader,
                                model=model,
                                device=device)
    
        train_loss_arr.append(train_loss)
        val_loss_arr.append(val_loss)
        # plot_graph(values=[train_loss_arr,val_loss_arr])
        if save_model:
            os.makedirs("models",exist_ok=True)
        if (epoch+1) % 2 == 0 :
            torch.save(model.state_dict(),os.path.join("models",f"model_{epoch+1}_{round(val_loss,2)}.pth"))
            print(f"model_{epoch+1}_{round(val_loss,2)}.pth saved !")
        
        print(f"\nEpoch: {epoch+1} train_loss: {round(train_loss,2)} val_loss: {round(val_loss,2)}")

    return (train_loss_arr,val_loss_arr)


class CustomDataset():
    def __init__(self,annotationFile:str,root_dir:str):
        self.coco = COCO(annotationFile)
        self.root_images = root_dir
        self.x_train = self.coco.getImgIds()
    def readFile(self,annotationfile:str):
        with open(annotationfile,"r") as f:
            return json.load(f)

    def __len__(self):
        return len(self.x_train)

    def get_resized_annotations(self,image_dim:tuple,bbox:list,resize:tuple):
        """resize annotations
        Args:
            image_dim (tuple): image shape
            bbox (list): bbox coordinates [x,y,w,h]
            resize (tuple): resize dimension (x,y)

        Returns:
            list: [x,y,w,h]
        """
        width_ratio,height_ratio = [resize[idx]/image_dim[idx] for idx in range(len(resize))]
        return (bbox[0]*height_ratio,bbox[1]*width_ratio,bbox[2]*height_ratio,bbox[3]*width_ratio)

    def __getitem__(self,index):
        img_info = self.coco.loadImgs(self.x_train[index])[0]
        img_main = cv2.imread(os.path.join(self.root_images, img_info['file_name']))
        img = cv2.resize(img_main,(600,600))
        # img = img_main
        img = img/255
        annIds = self.coco.getAnnIds(imgIds=img_info['id'])
        anns = self.coco.loadAnns(annIds)
        num_objs = len(anns)
        boxes = torch.zeros([num_objs,4], dtype=torch.float32)
        labels = torch.zeros(num_objs, dtype=torch.int64)

        for i in range(num_objs):
            [x,y,w,h]=anns[i]["bbox"]
            [x,y,w,h] = self.get_resized_annotations(img_main.shape,[x,y,w,h],(600,600))
            boxes[i] = torch.tensor(list(map(int,(x,y,x+w,y+h))))
            labels[i] = torch.tensor([anns[i]["category_id"]])

        img = torch.as_tensor(img, dtype=torch.float32)
        img = img.permute(2,0,1)
        data = {}
        data["boxes"] = boxes
        data["labels"] = labels
        return img, data
    def collate_fn(self,batch):
        return tuple(zip(*batch))
 

def plot_graph(values:list,
               label:str=["train loss","val loss"]):
    plt.subplot(2, 1, 1)
    plt.plot([x for x in range(1,len(values[0])+1)],values[0],color="green")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title(label=label[0])

    plt.subplot(2, 1, 2)
    plt.plot([x for x in range(1,len(values[1])+1)],values[1],color="green")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title(label=label[1])
    plt.savefig("graph.png")