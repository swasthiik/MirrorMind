# src/face_gaze_tracker.py

import cv2
import time

class FaceGazeTracker:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        self.last_blink_time = time.time()
    
    def track_gaze(self):
        cap = cv2.VideoCapture(0)
        gaze_info = {
            "face_detected": False,
            "eyes_detected": False,
            "blink_detected": False,
        }

        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            gaze_info["face_detected"] = len(faces) > 0
            gaze_info["eyes_detected"] = False
            gaze_info["blink_detected"] = False

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                gaze_info["eyes_detected"] = len(eyes) >= 2

                # Blink detection
                if len(eyes) == 0:
                    if time.time() - self.last_blink_time > 0.5:
                        gaze_info["blink_detected"] = True
                        self.last_blink_time = time.time()

                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 0, 0), 2)

                break  # Only process first face

            cv2.imshow("Face & Eye Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return gaze_info
