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
                    # Iris (eyes)
                    right_iris = face_landmarks.landmark[468]
                    left_iris = face_landmarks.landmark[473]
                    right_center = (int(right_iris.x * w), int(right_iris.y * h))
                    left_center = (int(left_iris.x * w), int(left_iris.y * h))
                    eyes_detected = True

                    # Smile check
                    top_lip = face_landmarks.landmark[13]
                    bottom_lip = face_landmarks.landmark[14]
                    lip_distance = abs(top_lip.y - bottom_lip.y)

                    if lip_distance > 0.03:  # Tune this value if needed
                        smile_detected = True

                    # Draw eyes
                    self._draw_robo_eye(frame, right_center)
                    self._draw_robo_eye(frame, left_center)

                    gaze_data.append((right_center, left_center))

                except IndexError:
                    continue

        # Overlay (You can show live loop here if needed)
        duration = int(time.time() - start_time)
        loop_status = "ComparisonLoop" if smile_detected else loop_type
        self._put_overlay_text(frame, loop_status, duration, break_suggested, pattern, context)

        # Show frame
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
