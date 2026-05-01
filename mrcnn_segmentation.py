import torchvision.transforms as transforms
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from glob import glob
import platform
from MaskRCNNDataset import MyDataset
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.amp import GradScaler
from tools.engine import train_one_epoch, evaluate, test_one_epoch
from torchvision.transforms import v2 as T
from PIL import Image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from tools.utils import collate_fn
from torchvision import tv_tensors

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomRotation(degrees=(-5,5)))
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip(0.5))
        transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2))

    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.Resize((512,512)))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

def main(args):    

    n_classes = 2

    train_dataset = MyDataset(dataset_dir=os.path.join("annotations", "train"), transforms=get_transform(train=True))
    test_dataset = MyDataset(dataset_dir=os.path.join("annotations", "test"), transforms=get_transform(train=False))

    batch_size = args.batch_size
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # if platform.system() == "Darwin":   # for macOS
    #     device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # else:   # Windows or Linux
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device('cpu')

    # load an instance segmentation model pre-trained on COCO
    model = maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes+1) # num_classes + background
    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, n_classes+1)
    model = model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.Adam(params, lr=0.005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    num_epochs = args.num_epochs

    scaler = GradScaler()
    
    train_losses = []
    test_losses = []
    best_test_loss = np.inf
    if args.command=='train':
        for epoch in range(num_epochs):
            print(f"Epoch: {epoch} / {num_epochs-1}")

            # train for one epoch, printing every 10 iterations
            train_logger, train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, scaler=scaler)
            test_logger, test_loss = test_one_epoch(model, test_loader, device, epoch)
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)

            print(f"Train Loss: {train_loss:.4f}\tTest Loss: {test_loss:.4f}")

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                os.makedirs('output', exist_ok=True)
                torch.save(model.state_dict(), os.path.join('output', 'model_best.pth'))

            # update the learning rate
            current_lr = lr_scheduler.get_last_lr()[0]
            print("current_lr:", current_lr)
            lr_scheduler.step()
            # evaluate on the test dataset
            # evaluate(model, test_loader, device=device)

            save_loss_curve(train_losses, test_losses)

    if args.command=='test':
        print("This is test mode.")

        # load model
        state_dict = torch.load(os.path.join('output', 'model_best.pth'), weights_only=True, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

        image_paths = glob(os.path.join('sample_dataset', 'test', '*.bmp'))
        idx = 0

        eval_transform = get_transform(train=False)

        while True:
            image_path = image_paths[idx]
            # image = cv2.imread(image_path)
            image = Image.open(image_path)
            image = tv_tensors.Image(image)            
            image = eval_transform(image)
            model.eval()
            with torch.no_grad():
                predictions = model([image.to(device)])
            pred = predictions[0]
            mask = (pred['masks'][0][0].cpu().numpy()*255).astype(np.uint8)
            cv2.imshow('mask', mask)
            cv2.waitKey()

            pred_labels = [f"{label}: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
            pred_boxes = pred["boxes"].long()

            output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")
            masks = (pred["masks"] > 0.7).squeeze(1)
            output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="green")
            output_image = output_image.permute(1, 2, 0).numpy()    #[c, h, w] --> [h, w, c]

            cv2.imshow('output_image', output_image)
            key = cv2.waitKey()
            if key==ord('q'):
                break
            elif key==ord('a'):
                idx -= 1
            elif key==ord('d'):
                idx += 1
            if idx < 0:
                idx = 0
            if idx > len(image_paths)-1:
                idx = len(image_paths)

def save_loss_curve(train_losses, test_losses):
    plt.plot(train_losses, label='train loss', marker='.')
    plt.plot(test_losses, label='test loss', marker='.')
    plt.grid()
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title("Loss Curve")
    plt.savefig("loss_curve.png")
    plt.close()



"""parsing and configuration"""
def argparse_args():  
    desc = "Pytorch implementation of 'Mask R-CNN Image Segmentation'"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('command', help="'train' or 'test' or 'labeling'")
    parser.add_argument('--num_epochs', default=300, type=int, help="The number of epochs to run")
    parser.add_argument('--batch_size', default=4, type=int, help="The number of mini-batchs for each epoch")
    return parser.parse_args()

if __name__ == '__main__':
    # parse arguments
    args = argparse_args()    
    if args is None:
        exit()
    print(args)
    
    main(args)