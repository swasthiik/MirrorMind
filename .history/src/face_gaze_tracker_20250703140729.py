# ✅ src/face_gaze_tracker.py

import cv2
import mediapipe as mp
import time
import numpy as np

class FaceGazeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.final_gaze_info = {}

    def stream_gaze_overlay(self, loop_detector, max_duration=120):
        cap = cv2.VideoCapture(0)
        start_time = time.time()
        gaze_data = []
        loop_type = "Normal"
        pattern = "DetectedGazePattern"
        context = "AutoDetect"
        break_suggested = "No"

        blink_counter = 0
        prev_eye_dist = None
        blink_detected = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            face_detected = False
            eyes_detected = False
            blink_detected = False

            if results.multi_face_landmarks:
                face_detected = True
                landmarks = results.multi_face_landmarks[0].landmark

                try:
                    left_eye = landmarks[474]
                    right_eye = landmarks[469]

                    left_center = (int(left_eye.x * w), int(left_eye.y * h))
                    right_center = (int(right_eye.x * w), int(right_eye.y * h))

                    gaze_data.append((right_center, left_center))
                    eyes_detected = True

                    # Draw eyes
                    self._draw_robo_eye(frame, left_center)
                    self._draw_robo_eye(frame, right_center)

                    # Blink Detection
                    eye_dist = np.linalg.norm(np.array(left_center) - np.array(right_center))
                    if prev_eye_dist is not None and eye_dist < 20:
                        blink_counter += 1
                    else:
                        blink_counter = 0
                    prev_eye_dist = eye_dist

                    if blink_counter > 2:
                        blink_detected = True

                except IndexError:
                    eyes_detected = False

            # Duration
            duration = int(time.time() - start_time)

            # 🔁 Loop Type Classification
            gaze_info = {
                "gaze_data": gaze_data,
                "face_detected": face_detected,
                "eyes_detected": eyes_detected,
                "blink_detected": blink_detected
            }

            loop_type = loop_detector.classify_loop(gaze_info, duration)
            break_suggested = "Yes" if loop_type in ["EscapeLoop", "FreezeLoop"] else "No"

            # 🖼️ Overlay
            self._put_overlay_text(frame, loop_type, duration, break_suggested, pattern, context)

            # 🔴 Show live feed
            cv2.imshow("🧠 MirrorMind – Real-Time Feed", frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Save final gaze info for logging
        self.final_gaze_info = {
            "gaze_data": gaze_data,
            "face_detected": face_detected,
            "eyes_detected": eyes_detected,
            "blink_detected": blink_detected,
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
        cv2.putText(frame, f"Loop: {loop_type}", (10, 30), font, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Duration: {duration}s", (10, 60), font, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Break: {break_suggested}", (10, 90), font, 0.7, (0, 0, 255) if break_suggested == "Yes" else (0, 255, 0), 2)
        cv2.putText(frame, f"Pattern: {pattern}", (10, 120), font, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Context: {context}", (10, 150), font, 0.7, (255, 128, 0), 2)
