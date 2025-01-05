import os
import cv2
import numpy as np
from PIL import Image
import json

# Paths
haarcascade_path = r"haarcascade_frontalface_default.xml"
train_image_path = r"train_images"  # Folder containing training images
model_save_path = r"TrainingImageLabel/Trainner.yml"
metadata_path = r"TrainingImageLabel/metadata.json"

# Create directories if they don't exist
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# Initialize recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Validate Haar cascade
if not os.path.exists(haarcascade_path):
    print("Haar Cascade file not found")
    exit()

detector = cv2.CascadeClassifier(haarcascade_path)

# Function to extract images and labels
def get_images_and_labels(path):
    faces = []
    ids = []
    metadata = {}  # Dictionary to store ID to name and age mapping

    for subdir in os.listdir(path):
        subdir_path = os.path.join(path, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for file in os.listdir(subdir_path):
            image_path = os.path.join(subdir_path, file)
            try:
                pil_image = Image.open(image_path).convert("L")  # Convert to grayscale
                image_np = np.array(pil_image, "uint8")

                # Extract metadata from filename (Format: name_age_ID.jpg)
                filename = os.path.splitext(file)[0]
                parts = filename.split("_")
                if len(parts) != 3:
                    print(f"Invalid file format: {file}. Skipping...")
                    continue

                name, age, person_id = parts[0], int(parts[1]), int(parts[2])

                # Detect face in the image
                faces_detected = detector.detectMultiScale(image_np)
                if len(faces_detected) == 0:
                    print(f"No face detected in {file}. Skipping...")
                    continue

                for (x, y, w, h) in faces_detected:
                    face = image_np[y:y+h, x:x+w]
                    faces.append(cv2.resize(face, (100, 100)))  # Resize to consistent size
                    ids.append(person_id)

                    # Add to metadata only if the person_id is not already present
                    if person_id not in metadata:
                        metadata[person_id] = {"name": name, "age": age}
                    break  # Use the first detected face only

            except Exception as e:
                print(f"Skipping file {file}: {e}")

    return faces, ids, metadata

# Get images, labels, and metadata
faces, ids, metadata = get_images_and_labels(train_image_path)

# Train recognizer
if len(faces) == 0:
    print("No training data found. Exiting...")
    exit()

recognizer.train(faces, np.array(ids))

# Save trained model
recognizer.save(model_save_path)
print(f"Model saved to {model_save_path}")

# Save metadata to JSON
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)
print(f"Metadata saved to {metadata_path}")
