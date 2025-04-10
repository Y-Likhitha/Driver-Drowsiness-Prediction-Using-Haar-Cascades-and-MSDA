import cv2
import numpy as np
from gtts import gTTS
import pygame
import time
import os

# --- INIT AUDIO FILES ---
ALERT_FILE = "alert_drowsy.mp3"
PERSON_ALERT_FILE = "alert_person.mp3"
VEHICLE_ALERT_FILE = "alert_vehicle.mp3"

if not os.path.exists(ALERT_FILE):
    tts = gTTS("Wake up! You are drowsy!", lang='en')
    tts.save(ALERT_FILE)

if not os.path.exists(PERSON_ALERT_FILE):
    tts = gTTS("Person is too close to the vehicle!", lang='en')
    tts.save(PERSON_ALERT_FILE)

if not os.path.exists(VEHICLE_ALERT_FILE):
    tts = gTTS("Vehicle is too close to you!", lang='en')
    tts.save(VEHICLE_ALERT_FILE)

# --- INIT PYGAME AUDIO ---
pygame.mixer.init()
drowsy_sound = pygame.mixer.Sound(ALERT_FILE)
person_sound = pygame.mixer.Sound(PERSON_ALERT_FILE)
vehicle_sound = pygame.mixer.Sound(VEHICLE_ALERT_FILE)

drowsy_channel = pygame.mixer.Channel(1)
person_channel = pygame.mixer.Channel(2)
vehicle_channel = pygame.mixer.Channel(3)

# --- Load Classifiers ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

# --- Load YOLO ---
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# --- Video Sources ---
internal_cam = cv2.VideoCapture(0)
external_cam = cv2.VideoCapture("http://192.168.43.1:4747/video")
if not external_cam.isOpened():
    external_cam = cv2.VideoCapture(1)

external_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
external_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Drowsiness tracking
prev_gray = None
prev_eye_points = None
drowsy_counter = 0
alert_threshold = 20

# --- Alert functions ---
def speak_alert(flag_name):
    if flag_name == "drowsy" and not drowsy_channel.get_busy():
        drowsy_channel.play(drowsy_sound)
    elif flag_name == "person" and not person_channel.get_busy():
        person_channel.play(person_sound)
    elif flag_name == "vehicle" and not vehicle_channel.get_busy():
        vehicle_channel.play(vehicle_sound)

def stop_alert(flag_name):
    if flag_name == "drowsy" and drowsy_channel.get_busy():
        drowsy_channel.stop()
    elif flag_name == "person" and person_channel.get_busy():
        person_channel.stop()
    elif flag_name == "vehicle" and vehicle_channel.get_busy():
        vehicle_channel.stop()

# Optical Flow params
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

while True:
    # --- Read internal cam (drowsiness)
    ret1, frame1 = internal_cam.read()
    if not ret1:
        continue
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    eyes_detected = False
    mouth_detected = False

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame1[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10, minSize=(15, 15))
        for (ex, ey, ew, eh) in eyes:
            eyes_detected = True
            eye_center = np.array([[ex + ew // 2, ey + eh // 2]], dtype=np.float32)
            if prev_gray is not None and prev_eye_points is not None:
                new_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_eye_points, None, **lk_params)
                drift = np.linalg.norm(new_points - prev_eye_points)
                if drift < 1:
                    drowsy_counter += 1
            prev_eye_points = eye_center

        mouths = mouth_cascade.detectMultiScale(roi_gray, 1.1, 8, minSize=(30, 30))
        for (mx, my, mw, mh) in mouths:
            if mh > h * 0.3:
                mouth_detected = True
                drowsy_counter += 2

    prev_gray = gray.copy()

    if not eyes_detected and not mouth_detected:
        drowsy_counter += 1
    else:
        drowsy_counter = max(0, drowsy_counter - 2)

    if drowsy_counter > alert_threshold:
        cv2.putText(frame1, "DROWSY ALERT!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        speak_alert("drowsy")
    else:
        stop_alert("drowsy")

    # --- Read external cam (object detection)
    for _ in range(5):
        external_cam.grab()
    ret2, frame2 = external_cam.read()
    if not ret2:
        continue

    height, width, _ = frame2.shape
    blob = cv2.dnn.blobFromImage(frame2, 0.00392, (320, 320), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    person_detected_close = False
    vehicle_detected_close = False

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x, center_y, w, h = (detection[:4] * np.array([width, height, width, height])).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                label = classes[class_id]
                color = (0, 255, 0) if label == "car" else (0, 0, 255)
                cv2.rectangle(frame2, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame2, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if label == "person" and w * h > 10000:
                    person_detected_close = True

                if label in ["car", "truck", "bus"] and w * h > 15000:
                    vehicle_detected_close = True

    if person_detected_close:
        speak_alert("person")
    else:
        stop_alert("person")

    if vehicle_detected_close:
        speak_alert("vehicle")
    else:
        stop_alert("vehicle")

    # --- Show output
    cv2.imshow("Drowsiness Detection", frame1)
    cv2.imshow("Object Detection", frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

internal_cam.release()
external_cam.release()
cv2.destroyAllWindows()
pygame.mixer.quit()

# Clean up audio files
for file in [ALERT_FILE, PERSON_ALERT_FILE, VEHICLE_ALERT_FILE]:
    if os.path.exists(file):
        os.remove(file)
