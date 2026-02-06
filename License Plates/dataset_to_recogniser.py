import glob
import os
import time
from PIL import Image, ImageFile
from ultralytics import YOLO
import yaml

ImageFile.LOAD_TRUNCATED_IMAGES = True 

localSubPaths = ["test","train","valid"]
model = YOLO("runs/detect/train8/weights/best.pt")
outDir = "Recogniser"
MIN_CONFIDENCE = 0.5

def min(newValue, currentValue):
    return newValue < currentValue

def max(newValue, currentValue):
    return newValue > currentValue

def cropBB(image, BBList, width, height):
    xCenter = float(BBList[0])
    yCenter = float(BBList[1])
    BBWidth = float(BBList[2])
    BBHeight = float(BBList[3])

    xMin = int((xCenter - BBWidth/2) * width)
    xMax = int((xCenter + BBWidth/2) * width)
    yMin = int((yCenter - BBHeight/2) * height)
    yMax = int((yCenter + BBHeight/2) * height)

    return image.crop((xMin,yMin,xMax, yMax))

def cropPolygon(image, BBList, width, height):
    xMin = float(BBList[0])
    xMax = float(BBList[0])
    yMin = float(BBList[1])
    yMax = float(BBList[1])
    i = 2

    while i < len(BBList):
        currentX = float(BBList[i])
        currentY = float(BBList[i+1])
        if min(currentX, xMin):
            xMin = currentX
        if min(currentY, yMin):
            yMin = currentY
        if max(currentX, xMax):
            xMax = currentX
        if max(currentY, yMax):
            yMax = currentY
        i += 2

    return image.crop((int(xMin*width),int(yMin*height),int(xMax*width), int(yMax*height)))

def pathToName(pathName):
    if pathName.lower().endswith(('.jpg', '.png')):
        return pathName.split("\\")[-1][0:-4]
    elif pathName.lower().endswith(".jpeg"):
        return pathName.split("\\")[-1][0:-5]
    return ""

def cropLabelledImages(image, localpath, width, height,className):
    foundImage = []
    
    with open(f"{localpath}.txt","r") as f:
        string = f.read()
        imagesLocations = string.split("\n")
        for imageLoc in imagesLocations:
            imageLoc = imageLoc.split(" ")[1: ]
            if len(imageLoc) == 0: continue
            foundImage.append(cropBB(image, imageLoc,width,height) 
                               if len(imageLoc) == 4 else 
                               cropPolygon(image, imageLoc,width, height))
    
    SaveCreatedImage(foundImage, localpath.split("/")[1], localpath.split("/")[-1], className)
            
def SaveCreatedImage(listOfImages, subPath, imageName, className):
    for i in range(len(listOfImages)):
        imagePath = f"{outDir}/{subPath}/images/{imageName}{'' if i == 0 else i}.jpg"
        labelPath = f"{outDir}/{subPath}/labels/{imageName}{'' if i == 0 else i}.txt"
        os.makedirs(os.path.dirname(imagePath), exist_ok=True)
        os.makedirs(os.path.dirname(labelPath), exist_ok=True)

        listOfImages[i].save(imagePath)
        with open(labelPath, "w") as txtFile:
            txtFile.write(f"{className} 0.5 0.5 1 1")
            

def fromPaths(StartPath):
    all_paths = glob.iglob(f'{StartPath}/*')
    all_images = []
    all_labels = []
    for paths in all_paths:
        if os.path.isdir(paths):
            local_images, local_labels = fromPaths(paths) #gatherImagesWithLabels() if (paths == "images" and "labels" in all_paths) else fromPaths(paths)
            all_images.extend(local_images)
            all_labels.extend(local_labels)
        else:
            imageName = pathToName(paths)
            if(imageName == ""): continue

            try:
                im = Image.open(paths)
                results = model([im], verbose=False, conf=MIN_CONFIDENCE)
                count = 0
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        im1 = im.crop((x1,y1,x2,y2))
                        if(im1.height * im1.width >= 30):
                            all_images.append(im1)
                            all_labels.append(f"{imageName}{"" if count == 0 else count}")
                            count += 1
            except OSError as e:
                print(f"Skipping corrupt file: {paths} - {e}")
                continue
            except Exception as e:
                print(f"Error processing {paths}: {e}")
                continue
        
        if(len(all_images) >= 5000):
            break

    return all_images, all_labels

            

def dataset_converter(localPath, className):
    FoundSubpaths = False

    with open(f"{outDir}/data.yaml") as f:
        data = yaml.safe_load(f)

    classNumber = 0
    
    if(className in data["names"]):
        classNumber = data["names"].index(className)
    else:
        classNumber = data["nc"]
        data["nc"] += 1
        data["names"].append(className)
        with open(f"{outDir}/data.yaml", 'w') as f:
            yaml.dump(data, f, default_flow_style=None, sort_keys=False)



    for subpath in localSubPaths:
        for imagePaths in glob.iglob(f'{localPath}/{subpath}/images/*'):
            FoundSubpaths = True
            imageName = pathToName(imagePaths)
            if(imageName == ""): continue

            im = Image.open(imagePaths)
            width = im.width
            height = im.height
            try:
                cropLabelledImages(im, f"{localPath}/{subpath}/labels/{imageName}",
                                   width, height, classNumber)
            except Exception as e:
                results = model([im])
                foundimages = []

                for result in results:
                    for box in result.boxes:
                        conf = box.conf[0].item()
                        if conf < MIN_CONFIDENCE: continue

                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        im1 = im.crop((x1,y1,x2,y2))
                        foundimages.append(im1)
                
                SaveCreatedImage(foundimages,subpath,imageName,className)
    
    if not FoundSubpaths:
        images, labels = fromPaths(localPath)
        n = len(images)
        split_1 = int(n * 0.6)
        split_2 = int(n * 0.8)

        for i in range(len(images)):
            subPath = ""
            if(i < split_1):
                subPath = "train"
            elif(i > split_2):
                subPath = "test"
            else:
                subPath = "valid"
            imagePath = f"{outDir}/{subPath}/images/{labels[i]}{'' if i == 0 else i}.jpg"
            labelPath = f"{outDir}/{subPath}/labels/{labels[i]}{'' if i == 0 else i}.txt"
            os.makedirs(os.path.dirname(imagePath), exist_ok=True)
            os.makedirs(os.path.dirname(labelPath), exist_ok=True)

            images[i].save(imagePath)
            with open(labelPath, "w") as txtFile:
                txtFile.write(f"{classNumber} 0.5 0.5 1 1")
        

#dataset_converter("canada","canada")

def masterFolderConverter(inputFolderPath):
    for folder in glob.iglob(f'{inputFolderPath}/*'):
        if os.path.isdir(folder):
            className = os.path.basename(folder) # safer than split('\\')
            
            print(f"Processing: {className}:")

            # 2. Record start time
            start_time = time.time()
            
            dataset_converter(folder, className)
            
            # 3. Record end time and calculate duration
            end_time = time.time()
            duration = end_time - start_time
            
            # 4. Print the result (rounded to 2 decimal places)
            print(f"Finished {className} in {duration:.2f} seconds")
            print("---")

masterFolderConverter("Countries")