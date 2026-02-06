from ultralytics import YOLO
import torch


def start_training():
    model = YOLO("yolo11n.pt") 
    model.train(
        data="Recogniser/data.yaml", 
        epochs=150, 
        imgsz=480, 
        device='cuda', 
        batch=-1, 
        workers=4,   
        optimizer='AdamW',
        patience=100, 
    )

if __name__ == '__main__':
    start_training()
