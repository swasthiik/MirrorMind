import cv2
import mediapipe as mp
import time
import math

class FaceGazeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
        self.drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

    def draw_robo_eye(self, frame, eye_center, radius=25):
        # Outer glowing ring
        cv2.circle(frame, eye_center, radius, (255, 255, 255), 2)
        # Inner glowing ring
        cv2.circle(frame, eye_center, int(radius / 2), (0, 255, 255), 1)
        # Scanning animation - rotating line
        angle = int(time.time() * 100) % 360
        angle_rad = angle * 3.14 / 180
        x_offset = int(math.cos(angle_rad) * radius)
        y_offset = int(math.sin(angle_rad) * radius)
        cv2.line(frame, eye_center,
                 (eye_center[0] + x_offset, eye_center[1] + y_offset),
                 (0, 255, 255), 2)

    def track_gaze(self):
        cap = cv2.VideoCapture(0)
        start_time = time.time()
        gaze_points = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Get right eye center (landmark 468 is eyeball center)
                    h, w, _ = frame.shape
                    cx = int(face_landmarks.landmark[468].x * w)
                    cy = int(face_landmarks.landmark[468].y * h)
                    gaze_points.append((cx, cy))

                    self.draw_robo_eye(frame, (cx, cy))

            # Display
            cv2.imshow("MirrorMind – Robo Eye Mode", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        return gaze_points
