# src/face_gaze_tracker.py

import cv2
import mediapipe as mp
import time

class FaceGazeTracker:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

    def draw_robo_eye(self, frame, center, radius=30, color=(0, 0, 255)):
        # Glowing Robo-Eye with LED color
        cv2.circle(frame, center, radius, color, 2)
        cv2.circle(frame, center, int(radius * 0.6), color, 1)
        cv2.circle(frame, center, int(radius * 0.3), (255, 255, 255), -1)
        cv2.circle(frame, center, 3, color, -1)
        return frame

    def get_gaze_direction(self, left_eye, right_eye):
        dx = right_eye.x - left_eye.x
        if dx < 0.06:
            return "Looking Left"
        elif dx > 0.08:
            return "Looking Right"
        else:
            return "Looking Center"

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

                    gaze_dir = self.get_gaze_direction(left_eye, right_eye)

                    # Choose eye LED color based on direction
                    if gaze_dir == "Looking Left":
                        eye_color = (255, 0, 0)  # 🔵 Blue
                    elif gaze_dir == "Looking Right":
                        eye_color = (0, 0, 255)  # 🔴 Red
                    else:
                        eye_color = (0, 255, 255)  # 🟡 Yellow

                    self.draw_robo_eye(frame, left_center, color=eye_color)
                    self.draw_robo_eye(frame, right_center, color=eye_color)

                    # Display gaze direction
                    cv2.putText(frame, f'Gaze: {gaze_dir}', (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color, 2)

            elapsed_time = int(time.time() - start_time)
            cv2.putText(frame, 'MirrorMind – Robo Eye Mode', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f'Duration: {elapsed_time}s', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            cv2.imshow("MirrorMind – Robo Eye Mode", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return "GazeData"

# Example usage
if __name__ == "__main__":
    gaze_tracker = FaceGazeTracker()
    gaze_tracker.track_gaze()
