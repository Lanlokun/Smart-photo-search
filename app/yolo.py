from ultralytics import YOLO
import os

_model = None

def get_model():
    global _model
    if _model is None:
        local_path = os.path.join(os.path.dirname(__file__), "yolov8n.pt")
        
        if os.path.exists(local_path):
            _model = YOLO(local_path)
        else:
            print("Local model not found, downloading 'yolov8n.pt' from Ultralytics hub...")
            _model = YOLO("yolov8n.pt")  
        
    return _model

def detect_objects(image_path):
    model = get_model()
    results = model(image_path)
    labels = set()
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            label = result.names[cls_id]
            labels.add(label)
    return list(labels)
