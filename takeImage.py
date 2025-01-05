import csv
import os
import cv2
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def text_to_speech(text):
    # A mock text-to-speech function for demonstration
    print(f"TTS: {text}")

def TakeImage(name, age, haar_cascade_path, train_image_path):
    if not age.strip() or not name.strip():
        t = "Please enter valid age and name."
        print(t)
        text_to_speech(t)
        return

    try:
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            raise Exception("Camera not accessible. Please check your webcam.")

        if not os.path.exists(haar_cascade_path):
            raise FileNotFoundError(f"Haar Cascade file not found at {haar_cascade_path}")
        detector = cv2.CascadeClassifier(haar_cascade_path)

        directory = f"{name}_{age}"
        path = os.path.join(train_image_path, directory)

        if not os.path.exists(path):
            os.makedirs(path)
            logging.info(f"Directory created: {path}")
        else:
            raise FileExistsError(f"Directory already exists for {directory}")

        sampleNum = 0
        while True:
            ret, img = cam.read()
            if not ret:
                raise Exception("Failed to read frame from camera.")

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                sampleNum += 1
                cv2.imwrite(
                    os.path.join(path, f"{name}_{age}_{sampleNum}.jpg"),
                    gray[y:y + h, x:x + w],
                )
                cv2.imshow("Frame", img)

            if cv2.waitKey(1) & 0xFF == ord("q") or sampleNum >= 50:
                break

        cam.release()
        cv2.destroyAllWindows()

        res = f"Images saved for Name: {name}, age: {age}"
        logging.info(res)
        print(res)
        text_to_speech(res)

    except FileExistsError as e:
        logging.error(str(e))
        print(str(e))
        text_to_speech(str(e))
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        print(f"Error: {str(e)}")
        text_to_speech(f"Error: {str(e)}")


if __name__ == "__main__":
    haar_cascade_path = "haarcascade_frontalface_default.xml"  # Path to Haar cascade file
    train_image_path = "train_images"  # Directory to save images

    print("Enter your details to proceed:")
    name = input("Enter your age: ").strip()
    age = input("Enter your name: ").strip()

    TakeImage(age, name, haar_cascade_path, train_image_path)
