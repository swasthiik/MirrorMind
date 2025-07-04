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
            refine_landmarks=True
        )
        self.final_gaze_info = {}

    def stream_gaze_overlay(self, loop_detector, max_duration=60):
        cap = cv2.VideoCapture(0)
        start_time = time.time()
        gaze_data = []
        pattern = "DetectedGazePattern"
        context = "AutoDetect"

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
                        right_iris = face_landmarks.landmark[468]
                        left_iris = face_landmarks.landmark[473]
                        right_center = (int(right_iris.x * w), int(right_iris.y * h))
                        left_center = (int(left_iris.x * w), int(left_iris.y * h))
                        gaze_data.append((right_center, left_center))
                        self._draw_robo_eye(frame, right_center)
                        self._draw_robo_eye(frame, left_center)
                        eyes_detected = True

                        # Smile Detection
                        mouth_left = face_landmarks.landmark[61]
                        mouth_right = face_landmarks.landmark[291]
                        mouth_top = face_landmarks.landmark[13]
                        mouth_bottom = face_landmarks.landmark[14]

                        mouth_w = np.linalg.norm(np.array([mouth_left.x, mouth_left.y]) - np.array([mouth_right.x, mouth_right.y]))
                        mouth_h = np.linalg.norm(np.array([mouth_top.x, mouth_top.y]) - np.array([mouth_bottom.x, mouth_bottom.y]))

                        if mouth_w / mouth_h > 1.8:
                            smile_detected = True
                    except:
                        continue

            duration = int(time.time() - start_time)
            gaze_info = {
                "gaze_data": gaze_data,
                "face_detected": face_detected,
                "eyes_detected": eyes_detected,
                "blink_detected": False,
                "smile_detected": smile_detected
            }

            loop_type = loop_detector.classify_loop(gaze_info, duration)
            if smile_detected:
                loop_type = "ComparisonLoop"
            break_suggested = "Yes" if loop_type in ["EscapeLoop", "FreezeLoop"] else "No"

            self._put_overlay_text(frame, loop_type, duration, break_suggested, pattern, context)
            cv2.imshow("MirrorMind - Live Gaze Feed", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        self.final_gaze_info = gaze_info

    def _draw_robo_eye(self, frame, center, color=(255, 0, 0)):
        x, y = center
        for i in range(4):
            radius = 8 + i * 4
            opacity = 1 - i / 4
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
