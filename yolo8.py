import cv2
from ultralytics import YOLO
model = YOLO("yolov8n-face.onnx")  
cap = cv2.VideoCapture(0) 
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    frame=results[0].plot()
    cv2.imshow("Détection de visage YOLOv8", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
