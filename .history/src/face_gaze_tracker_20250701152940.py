def track_gaze(self):
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    gaze_info = {
        "face_detected": False,
        "eyes_detected": False,
        "scroll_detected": False,
    }

    output_frame = None  # Final frame to return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            gaze_info["face_detected"] = True
            gaze_info["eyes_detected"] = True

            for face_landmarks in results.multi_face_landmarks:
                try:
                    right_iris = face_landmarks.landmark[468]
                    left_iris = face_landmarks.landmark[473]
                    right_center = (int(right_iris.x * w), int(right_iris.y * h))
                    left_center = (int(left_iris.x * w), int(left_iris.y * h))

                    # Draw robo eye
                    self._draw_robo_eye(frame, right_center, (0, 180, 255))  # Cyan
                    self._draw_robo_eye(frame, left_center, (0, 180, 255))

                except IndexError:
                    pass

        output_frame = frame.copy()

        # 👁️ Break after 1 loop (just return frame to Streamlit)
        break

    cap.release()
    return gaze_info, output_frame
