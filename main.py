import cv2
import numpy as np
import json
import os
import pyttsx3

tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 150)

haarcascade_path = r"haarcascade_frontalface_default.xml"
model_load_path = r"TrainingImageLabel/Trainner.yml"
metadata_path = r"TrainingImageLabel/metadata.json"

if not os.path.exists(haarcascade_path):
    print("Haar Cascade file not found. Exiting...")
    exit()
face_cascade = cv2.CascadeClassifier(haarcascade_path)

# Load the trained face recognizer model
if not os.path.exists(model_load_path):
    print("Trained model not found. Exiting...")
    exit()
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_load_path)

# Load metadata
if not os.path.exists(metadata_path):
    print("Metadata file not found. Exiting...")
    exit()
with open(metadata_path, "r") as f:
    metadata = json.load(f)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error accessing the webcam. Exiting...")
    exit()

print("Press 'q' to exit the program.")

spoken_names = set()

while True:

    ret, frame = cap.read()
    if not ret:
        print("Error reading frame. Exiting...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:

        face = gray[y:y + h, x:x + w]

        person_id, confidence = recognizer.predict(cv2.resize(face, (100, 100)))

        if confidence < 100: 
            person_data = metadata.get(str(person_id), {})
            name = person_data.get("name", "Unknown")
            age = person_data.get("age", "Unknown")
            label = f"{name}, {age} yrs"
            
            if name != "Unknown" and name not in spoken_names:
                if name == "Arya Gangwar":
                    print(name)
                    tts_engine.say(f"Welcome Boss")
                    tts_engine.runAndWait() 
                spoken_names.add(name)
                              
        else:
            label = "Unknown"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
