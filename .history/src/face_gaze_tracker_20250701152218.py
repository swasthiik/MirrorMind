import cv2
import mediapipe as mp
import time

class FaceGazeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True  # Needed for iris detection
        )

    def track_gaze(self):
        cap = cv2.VideoCapture(0)
        start_time = time.time()
        gaze_data = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    try:
                        # Iris landmark index (468: right, 473: left)
                        right_iris = face_landmarks.landmark[468]
                        left_iris = face_landmarks.landmark[473]

                        right_center = (int(right_iris.x * w), int(right_iris.y * h))
                        left_center = (int(left_iris.x * w), int(left_iris.y * h))

                        gaze_data.append((right_center, left_center))

                        # 🔵 Draw cyber blue robo eye
                        self._draw_robo_eye(frame, right_center, (255, 100, 0))   # Light Blue
                        self._draw_robo_eye(frame, left_center, (0, 180, 255))     # Cyan Blue

                    except IndexError:
                        pass

            # Display window
            cv2.imshow("MirrorMind – Cyber Eye Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        
        return gaze_info
    
        duration = time.time() - start_time
        return gaze_data

    def _draw_robo_eye(self, frame, center, color=(0, 180, 255)):
        """Draw glowing blue cybernetic eye with concentric rings"""
        x, y = center
        glow_layers = 4
        base_radius = 8

        for i in range(glow_layers):
            radius = base_radius + i * 4
            opacity = 1 - i / glow_layers
            glow_color = tuple(int(c * opacity) for c in color)
            cv2.circle(frame, (x, y), radius, glow_color, 1)

        # Central glow (pupil)
        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
