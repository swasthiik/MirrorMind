import cv2
import time
import mediapipe as mp

class FaceGazeTracker:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        self.last_blink_time = time.time()

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1)
        self.mp_draw = mp.solutions.drawing_utils

    def track_gaze(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        gaze_info = {
            "face_detected": False,
            "eyes_detected": False,
            "blink_detected": False,
            "scroll_detected": False
        }

        prev_hand_y = None
        scroll_counter = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab camera frame!")
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            gaze_info["face_detected"] = len(faces) > 0
            gaze_info["eyes_detected"] = False
            gaze_info["blink_detected"] = False
            gaze_info["scroll_detected"] = False

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                gaze_info["eyes_detected"] = len(eyes) >= 2

                if len(eyes) == 0:
                    if time.time() - self.last_blink_time > 0.5:
                        gaze_info["blink_detected"] = True
                        self.last_blink_time = time.time()

                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 0, 0), 2)

                break  # Only process the first face

            # --- Hand Detection ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    y_positions = [lm.y for lm in hand_landmarks.landmark]
                    avg_y = sum(y_positions) / len(y_positions)

                    if prev_hand_y is not None and abs(avg_y - prev_hand_y) > 0.05:
                        scroll_counter += 1

                    prev_hand_y = avg_y

                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                if scroll_counter >= 3:
                    gaze_info["scroll_detected"] = True
                    scroll_counter = 0  # reset

            # Display
            cv2.putText(frame, f"Scroll: {gaze_info['scroll_detected']}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("MirrorMind – Gaze & Scroll Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return gaze_info
