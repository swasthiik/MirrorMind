def track_gaze(self, max_duration=10):
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    gaze_data = []
    final_frame = None

    face_detected = False
    eyes_detected = False
    blink_detected = False  # Simplified placeholder

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            face_detected = True

            for face_landmarks in results.multi_face_landmarks:
                try:
                    right_iris = face_landmarks.landmark[468]
                    left_iris = face_landmarks.landmark[473]

                    right_center = (int(right_iris.x * w), int(right_iris.y * h))
                    left_center = (int(left_iris.x * w), int(left_iris.y * h))

                    gaze_data.append((right_center, left_center))
                    self._draw_robo_eye(frame, right_center, (0, 180, 255))
                    self._draw_robo_eye(frame, left_center, (0, 180, 255))

                    eyes_detected = True

                except IndexError:
                    eyes_detected = False
                    continue

        final_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if time.time() - start_time > max_duration:
            break

    cap.release()
    
    return {
        "gaze_data": gaze_data,
        "frame": final_frame,
        "face_detected": face_detected,
        "eyes_detected": eyes_detected,
        "blink_detected": blink_detected  # Not implemented fully yet
    }
