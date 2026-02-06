from ultralytics import YOLO
import torch


def start_training():
    model = YOLO("yolo11n-cls.pt") 
    model.train(
        data="RecogniserV3", 
        epochs=75, 
        imgsz=224, 
        device='cuda', 
        batch=128, 
        workers=8,   
        optimizer='AdamW',
        patience=10, 
    )

if __name__ == '__main__':
    start_training()
