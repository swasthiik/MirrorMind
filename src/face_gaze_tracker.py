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

    def stream_gaze_overlay_live(self):
        cap = cv2.VideoCapture(0)
        start_time = time.time()
        gaze_data = []
        face_detected = False
        eyes_detected = False
        smile_detected = False
        blink_count = 0
        prev_eye_state = True  # Assume eyes open
        blink_start_time = 0

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
                        # Iris detection (eyes)
                        right_iris = face_landmarks.landmark[468]
                        left_iris = face_landmarks.landmark[473]
                        right_center = (int(right_iris.x * w), int(right_iris.y * h))
                        left_center = (int(left_iris.x * w), int(left_iris.y * h))
                        eyes_detected = True
                        gaze_data.append((right_center, left_center))

                        # Smile check
                        top_lip = face_landmarks.landmark[13]
                        bottom_lip = face_landmarks.landmark[14]
                        lip_distance = abs(top_lip.y - bottom_lip.y)
                        if lip_distance > 0.03:
                            smile_detected = True

                        # Blink detection (basic)
                        left_eye_top = face_landmarks.landmark[159]
                        left_eye_bottom = face_landmarks.landmark[145]
                        eye_distance = abs(left_eye_top.y - left_eye_bottom.y)
                        eye_closed = eye_distance < 0.01

                        if not eye_closed and not prev_eye_state:
                            blink_count += 1
                        prev_eye_state = not eye_closed

                        # Draw eyes
                        self._draw_robo_eye(frame, right_center)
                        self._draw_robo_eye(frame, left_center)

                    except IndexError:
                        continue

            duration = int(time.time() - start_time)
            loop_status = "ComparisonLoop" if smile_detected else "Normal"
            context = self._guess_context(gaze_data, blink_count, face_detected)

            self._put_overlay_text(frame, loop_status, duration, "No", "DetectedGazePattern", context)
            cv2.imshow("🧠 MirrorMind Live", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        context = self._guess_context(gaze_data, blink_count, face_detected)
        self.final_gaze_info = {
            "gaze_data": gaze_data,
            "face_detected": face_detected,
            "eyes_detected": eyes_detected,
            "blink_detected": blink_count > 3,
            "smile_detected": smile_detected,
            "context": context
        }

        return  # Fix for NoneType error when used in for-loop

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

    def _guess_context(self, gaze_data, blink_count, face_detected):
        if not face_detected:
            return "Escape/Idle"
        if blink_count > 5:
            return "Writing/Thinking"
        if len(gaze_data) < 2:
            return "Unknown"

        movement = 0
        prev = gaze_data[0]
        for point in gaze_data[1:]:
            dist = np.linalg.norm(np.array(point[0]) - np.array(prev[0])) + \
                   np.linalg.norm(np.array(point[1]) - np.array(prev[1]))
            movement += dist
            prev = point

        avg_movement = movement / len(gaze_data)

        if avg_movement < 5:
            return "Reading"
        elif avg_movement < 20:
            return "Writing"
        else:
            return "Browsing"
