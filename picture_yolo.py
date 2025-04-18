from ultralytics import YOLO
import cv2
import math 

# Load image
img = cv2.imread("/Users/qijianma/code/Hackathon---Team-7/ezgif-frame-002_1.jpg")  # <-- Replace with your image path

# Load YOLO model
model = YOLO("yolo11l.pt")

# Define class names
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush", "phone"]

# Run detection
results = model(img, stream=True, conf=0.3)

# Parse results
for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        confidence = math.ceil((box.conf[0]*100))/100
        cls = int(box.cls[0])
        org = (x1, y1)
        cv2.putText(img, f"{classNames[cls]} {confidence}", org,
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

# Show image
cv2.imshow('Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
