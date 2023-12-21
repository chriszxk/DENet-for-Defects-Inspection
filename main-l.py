from ultralytics import YOLO

# Load a model
model = YOLO('yolov8l.yaml')  # build a new model from YAML
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
model.train(data='/home/xunkuai/github/yolov8/ultralytics/datasets/hk.yaml', epochs=500, imgsz=1024, patience=0, workers=25, device=1, project="kilu069-l/")
