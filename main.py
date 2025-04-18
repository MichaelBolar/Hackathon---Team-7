import threading
import cv2
import speech_recognition as sr
import ollama
from gtts import gTTS
import os
from pathlib import Path
from ultralytics import YOLO
import math

# Shared state
current_frame = None
frame_lock   = threading.Lock()
detection_lock     = threading.Lock()
current_detections = []



def camera_loop():
    global current_frame,current_detections
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    model = YOLO("yolo11l.pt")
    classNames = ["person", 
              "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush","phone"]

    if not cap.isOpened():
        print("Failed to open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            continue
        
        results = model(frame, stream=False,conf=0.7, verbose = False)
        detections = []
        with frame_lock:
            current_frame = frame.copy()
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # put box in cam
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                # print("Confidence --->",confidence)

                # class name
                cls = int(box.cls[0])
                # print("Class name -->", classNames[cls])
                detections.append(classNames[cls])

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(frame, classNames[cls], org, font, fontScale, color, thickness)


        with detection_lock:
            if detections:
                current_detections = detections.copy()


        cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def audio_loop():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)

    while True:
        with mic as source:
            print("Listening…")
            audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print("You said:", text)
        except Exception:
            print("Couldn’t understand, retrying…")
            continue

        # Snapshot
        with frame_lock:
            if current_frame is not None:
                snap = current_frame.copy()  
            else:
                print("no frame is available")
                snap = None
        
        with detection_lock:
            if current_detections:
                yolo_key_word = "The model sees " + ','.join(map(str,current_detections)) + " in the picture"
                print (yolo_key_word)

        if yolo_key_word != "":
            systemtext = "There is a other model pre process this frame of picture." + str(yolo_key_word)+ "please use it as refrence to answer user's question: " + text
            print(text)

        images = []

        if snap is not None:
            img_path = Path("snapshot.jpg")
            cv2.imwrite(str(img_path), snap)
            images = [str(img_path)]

        try:
            resp = ollama.chat(
                model="blindaid",
            messages = [
            {"role": "system", "content": systemtext},
            {"role": "user", "content": text, "images": images}
            ]
            )["message"]["content"]
            print("Response:", resp)
        except Exception as e:
            print("error:", e)
            continue

        tts = gTTS(resp)
        tts_file = "response.mp3"
        tts.save(tts_file)
        if os.name == 'nt':
            os.system(f"start {tts_file}")
        else:
            os.system(f"afplay {tts_file}")

if __name__ == "__main__":
    t_audio = threading.Thread(target=audio_loop, daemon=True)
    t_audio.start()
    camera_loop()

   