from ultralytics import YOLO
import torch


def start_training():
    model = YOLO("yolo11n.pt") 
    model.train(
        data="LicensePlateDatasetV2/data.yaml", 
        epochs=50, 
        imgsz=512, 
        device='cuda', 
        batch=-1, 
        workers=8,   
        cache='disk',
        patience=10, 
    )

if __name__ == '__main__':
    start_training()
