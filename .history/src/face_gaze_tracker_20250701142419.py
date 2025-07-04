import cv2
import time
from src.loop_detector import LoopDetector

class FaceGazeTracker:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        self.last_blink_time = time.time()

    def track_gaze(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        start_time = time.time()

        gaze_info = {
            "face_detected": False,
            "eyes_detected": False,
            "blink_detected": False,
        }

        loop_type = "Detecting..."
        pattern = "DetectedGazePattern"
        context = "Working"
        break_suggested = "No"
        detector = LoopDetector()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            gaze_info["face_detected"] = len(faces) > 0
            gaze_info["eyes_detected"] = False
            gaze_info["blink_detected"] = False

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                gaze_info["eyes_detected"] = len(eyes) >= 2

                if len(eyes) == 0:
                    if time.time() - self.last_blink_time > 0.5:
                        gaze_info["blink_detected"] = True
                        self.last_blink_time = time.time()

                    cv2.putText(frame, "🔴 Eyes Not Detected", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                for (ex, ey, ew, eh) in eyes:
                    center_x = x + ex + ew // 2
                    center_y = y + ey + eh // 2
                    radius = int((ew + eh) / 3)

                    # 💠 Robot style glowing eye
                    cv2.circle(frame, (center_x, center_y), radius + 2, (255, 255, 255), 1)
                    cv2.circle(frame, (center_x, center_y), radius, (0, 255, 255), 2)
                    cv2.circle(frame, (center_x, center_y), radius - 5, (0, 150, 255), 1)
                    cv2.circle(frame, (center_x, center_y), 3, (255, 255, 255), -1)

                    # 💫 Animated sparkle
                    if int(time.time() * 10) % 2 == 0:
                        cv2.circle(frame, (center_x + 4, center_y - 3), 2, (255, 255, 255), -1)

                break  # only first face

            # 🔁 Loop detection
            duration = time.time() - start_time
            loop_type = detector.classify_loop(gaze_info, duration)
            break_suggested = "Yes" if loop_type in ["EscapeLoop", "FreezeLoop"] else "No"

            # 🧠 Overlay info with pro colors
            cv2.putText(frame, f"LoopType: {loop_type}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Duration: {int(duration)}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
            cv2.putText(frame, f"Break: {break_suggested}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 80, 80), 2)
            cv2.putText(frame, f"Pattern: {pattern}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
            cv2.putText(frame, f"Context: {context}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 255, 180), 2)

            cv2.imshow("MirrorMind – Loop Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return gaze_info
