from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image, ImageDraw, ImageFont
from tkinter import filedialog
from ultralytics import YOLO
import glob
import os

detectModel = YOLO("runs/detect/train12/weights/best.pt")
recognitionModel = YOLO("runs/classify/train5/weights/best.pt")


def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename

def open_img():
    filename = openfn()
    if filename:
        imgTemp = Image.open(filename)
        img_final = imgTemp.copy()
        draw = ImageDraw.Draw(img_final)
        imgTemp.thumbnail((400, 400))
        img = ImageTk.PhotoImage(imgTemp)
        panel_input.configure(image=img)
        panel_input.image = img
        result = detectModel(filename)[0]
        
        plotted_array_bgr = result.plot()
        plotted_array_rgb = plotted_array_bgr[:, :, ::-1]
        processed_pil = Image.fromarray(plotted_array_rgb)
        processed_pil.thumbnail((400, 400))
        processed_tk = ImageTk.PhotoImage(processed_pil)
        panel_processed.configure(image=processed_tk)
        panel_processed.image = processed_tk

        Width, Height = img_final.size
        print(Width,Height)
        scale = Height / 1000
        scale = max(scale, 1.5) 
        line_thickness = int(3 * scale)
        font_size = int(25 * scale)

        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        for box in result.boxes:
            lp_x1, lp_y1, lp_x2, lp_y2 = map(int, box.xyxy[0].tolist())
            im1 = img_final.crop((lp_x1, lp_y1, lp_x2, lp_y2))
            res = recognitionModel(im1)[0]
            top_class_id = res.probs.top1

            name = res.names[top_class_id]
            conf = res.probs.top1conf.item()
            label_text = f"{name} {conf:.2f}"

            draw.rectangle([lp_x1, lp_y1, lp_x2, lp_y2], outline="red", width=line_thickness)
            
            char_width = font_size * 0.6 
            text_width = len(label_text) * char_width
            text_height = font_size * 1.2
            draw.rectangle([lp_x1, lp_y1 - text_height, lp_x1 + text_width, lp_y1], fill="red")
            
            draw.text(
                (lp_x1 + (5 * scale), lp_y1 - text_height),
                label_text, 
                fill="white", 
                font=font
             )
        img_final.thumbnail((400, 400))
        final_tk = ImageTk.PhotoImage(img_final)
        panel_recognition.configure(image=final_tk)
        panel_recognition.image = final_tk

root = Tk(screenName="License_Plate_test", baseName="License_Plate_test")
root.state("zoomed")
root.geometry("500x400+0+0")
root.title("License_Plate_test")
root.resizable(width=True, height=True)

root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

mainframe = ttk.Frame(root, padding="10")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))

img = ""
button = Button(mainframe, text="Load Image",width="30", command=open_img)
button.grid(column=0, row=0, columnspan=3, pady=(0, 20),sticky=(E,W))

lbl_input = ttk.Label(mainframe, text="Input Image:")
lbl_input.grid(column=0, row=1, sticky=W)

lbl_processed = ttk.Label(mainframe, text="Image with detection:")
lbl_processed.grid(column=1, row=1, sticky=W)

lbl_recognition = ttk.Label(mainframe, text="Image with recognition:")
lbl_recognition.grid(column=2, row=1, sticky=W)

panel_input = Label(mainframe, bg="gray") # bg="gray" just to show the empty space
panel_input.grid(column=0, row=2, sticky=(N, W, E, S), padx=5)

panel_processed = Label(mainframe, bg="lightgray")
panel_processed.grid(column=1, row=2, sticky=(N, W, E, S), padx=5)

panel_recognition = Label(mainframe, bg="green")
panel_recognition.grid(column=2, row=2, sticky=(N, W, E, S), padx=5)


mainframe.columnconfigure(0, weight=1)
mainframe.columnconfigure(1, weight=1)
mainframe.columnconfigure(2, weight=1)
mainframe.rowconfigure(2, weight=1)
for child in mainframe.winfo_children(): 
    child.grid_configure(padx=5, pady=5)


root.mainloop()