import cv2
import mediapipe as mp
import time
import numpy as np

class FaceGazeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        self.face_detector = self.mp_face_detection.FaceDetection(min_detection_confidence=0.6)
        self.mp_drawing = mp.solutions.drawing_utils
        self.final_gaze_info = {}

    def stream_gaze_overlay(self, loop_type="Normal", pattern="DetectedGazePattern", context="AutoDetect", break_suggested="No", max_duration=None):
        cap = cv2.VideoCapture(0)
        start_time = time.time()
        gaze_data = []
        face_detected = False
        eyes_detected = False
        smile_detected = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            face_detected = False
            eyes_detected = False
            smile_detected = False

            if results.multi_face_landmarks:
                face_detected = True
                for face_landmarks in results.multi_face_landmarks:
                    try:
                        # Eyes detection
                        right_iris = face_landmarks.landmark[468]
                        left_iris = face_landmarks.landmark[473]
                        right_center = (int(right_iris.x * w), int(right_iris.y * h))
                        left_center = (int(left_iris.x * w), int(left_iris.y * h))
                        eyes_detected = True

                        # Smile detection
                        top_lip = face_landmarks.landmark[13]
                        bottom_lip = face_landmarks.landmark[14]
                        lip_distance = abs(top_lip.y - bottom_lip.y)

                        if lip_distance > 0.03:
                            smile_detected = True

                        # Draw robo eyes
                        self._draw_robo_eye(frame, right_center)
                        self._draw_robo_eye(frame, left_center)
                        gaze_data.append((right_center, left_center))

                    except IndexError:
                        continue

            # Update loop type in real time
            current_loop = "ComparisonLoop" if smile_detected else loop_type

            duration = int(time.time() - start_time)
            self._put_overlay_text(frame, current_loop, duration, break_suggested, pattern, context)

            # Show live feed
            cv2.imshow("🧠 MirrorMind Live", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        self.final_gaze_info = {
            "gaze_data": gaze_data,
            "face_detected": face_detected,
            "eyes_detected": eyes_detected,
            "blink_detected": False,
            "smile_detected": smile_detected,
            "context": context
        }

    def _draw_robo_eye(self, frame, center, color=(255, 0, 0)):
        x, y = center
        glow_layers = 4
        base_radius = 8

        for i in range(glow_layers):
            radius = base_radius + i * 4
            opacity = 1 - i / glow_layers
            glow_color = tuple(int(c * opacity) for c in color)
            cv2.circle(frame, (x, y), radius, glow_color, 1)

        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)

    def _put_overlay_text(self, frame, loop_type, duration, break_suggested, pattern, context):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"LoopType: {loop_type}", (10, 30), font, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Duration: {duration}s", (10, 60), font, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Break: {break_suggested}", (10, 90), font, 0.7, (0, 0, 255) if break_suggested == "Yes" else (0, 255, 0), 2)
        cv2.putText(frame, f"Pattern: {pattern}", (10, 120), font, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Context: {context}", (10, 150), font, 0.7, (255, 128, 0), 2)
