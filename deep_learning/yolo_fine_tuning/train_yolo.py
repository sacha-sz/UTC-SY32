from ultralytics import YOLO

yolo = YOLO('yolov8n.pt')
yolo.train(data='config.yaml', epochs=3, imgsz=640)
valid_results = yolo.val()
print(valid_results)

