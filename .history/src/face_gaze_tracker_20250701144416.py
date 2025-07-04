# src/face_gaze_tracker.py

import cv2
import mediapipe as mp
import time

class FaceGazeTracker:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

    def draw_robo_eye(self, frame, center, radius=30):
        # Glowing Robo-Eye Overlay
        cv2.circle(frame, center, radius, (0, 255, 255), 2)         # Outer glow
        cv2.circle(frame, center, int(radius * 0.6), (0, 200, 255), 2)
        cv2.circle(frame, center, int(radius * 0.3), (255, 255, 255), -1)  # Center white core
        cv2.circle(frame, center, 3, (0, 0, 255), -1)               # Red dot center
        return frame

    def track_gaze(self):
        cap = cv2.VideoCapture(0)
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    left_eye = face_landmarks.landmark[468]
                    right_eye = face_landmarks.landmark[473]

                    left_center = (int(left_eye.x * w), int(left_eye.y * h))
                    right_center = (int(right_eye.x * w), int(right_eye.y * h))

                    # 👁️ Draw Robo Eye
                    self.draw_robo_eye(frame, left_center)
                    self.draw_robo_eye(frame, right_center)

            # 🧠 Info
            elapsed_time = int(time.time() - start_time)
            cv2.putText(frame, f'MirrorMind – Robo Eye Mode', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(frame, f'Duration: {elapsed_time}s', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            cv2.imshow("MirrorMind – Robo Eye Mode", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return "GazeData"  # Placeholder for future logic
