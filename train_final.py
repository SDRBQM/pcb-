from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8s.yaml')  # build a new model from YAML
model = YOLO('ultralytics/cfg/models/v8/yolov8_one_swinTrans.yaml')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8s.yaml').load('yolov8s.pt')  # build from YAML and transfer weights

# Train the model
if __name__ == '__main__':
    model.train(data=r'E:\deskdop\1\1\NEU-DET-with-yolov8-main\data\data.yaml', pretrained='yolov8n.pt', epochs=400,batch=2, imgsz=640, device='0')

