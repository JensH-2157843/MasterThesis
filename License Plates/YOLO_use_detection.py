from ultralytics import YOLO
import glob
import os

model = YOLO("runs/detect/train3/weights/last.pt")
countries = ["Bahrain", "Ireland", "Norway", "Pakistan", "USA","Brazil","Estonia","Finland","Kazakhstan","Lithuania","Serbia","UAE","Belgium","France","Germany","Hungary","Italy","Netherlands","Poland","Spain","UK"]

for country in countries:
    folder_dir = f'Places/{country}'
    output_dir = f'results_3L/{country}'

    plates = []

    os.makedirs(output_dir, exist_ok=True)

    for image_path in glob.iglob(f'{folder_dir}/*'):

        if image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            plates.append(image_path)

    results = model(plates) 

    for result in results:
        boxes = result.boxes  
        masks = result.masks  
        keypoints = result.keypoints  
        probs = result.probs  
        obb = result.obb  
        filename = os.path.basename(result.path)
        save_path = os.path.join(output_dir, filename)
        result.save(filename=save_path)  