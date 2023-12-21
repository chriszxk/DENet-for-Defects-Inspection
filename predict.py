from ultralytics import YOLO
#load a pretrained YOLOv8n model
model=YOLO('/home/xunkuai/crack_detection/AS/AS/m/06/train4/weights/best.pt')

#Define path to the image file
source = '/home/xunkuai/github/yolov5/hk/images/test2017/'

model.predict(source, save=True, imgsz=1024, conf=0.5, iou=0.7, show_conf=True, visualize=False)