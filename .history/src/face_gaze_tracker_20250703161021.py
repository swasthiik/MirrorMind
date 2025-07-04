import cv2
import mediapipe as mp
import time
import numpy as np

class FaceGazeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        self.final_gaze_info = {}

    def detect_context_from_gaze(self, gaze_data):
        if not gaze_data:
            return "Unknown"
        if len(gaze_data) > 20:
            return "Reading"
        elif len(gaze_data) > 10:
            return "Writing"
        else:
            return "Browsing"

    def stream_gaze_overlay(self, loop_type="Normal", pattern="DetectedGazePattern", context="AutoDetect", break_suggested="No", max_duration=120):
        cap = cv2.VideoCapture(0)
        start_time = time.time()
        gaze_data = []
        face_detected = False
        eyes_detected = False
        blink_detected = False
        smile_detected = False
        prev_eye_y = None
        blink_counter = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            face_results = self.face_detector.process(rgb_frame)

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

                        current_eye_y = left_center[1]
                        if prev_eye_y and abs(prev_eye_y - current_eye_y) > 6:
                            blink_counter += 1
                        prev_eye_y = current_eye_y

                    except IndexError:
                        continue

            if face_results.detections:
                for detection in face_results.detections:
                    if detection.score[0] > 0.6:
                        smile_detected = True

            inferred_context = self.detect_context_from_gaze(gaze_data)
            duration = int(time.time() - start_time)
            self._put_overlay_text(frame, loop_type, duration, break_suggested, pattern, inferred_context)
            cv2.imshow("🧠 MirrorMind Live Feed", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        self.final_gaze_info = {
            "gaze_data": gaze_data,
            "face_detected": face_detected,
            "eyes_detected": eyes_detected,
            "blink_detected": blink_counter > 2,
            "smile_detected": smile_detected,
            "context": inferred_context
        }

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
