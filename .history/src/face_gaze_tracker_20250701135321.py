def track_gaze(self):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    start_time = time.time()

    gaze_info = {
        "face_detected": False,
        "eyes_detected": False,
        "blink_detected": False,
    }

    loop_type = "Detecting..."
    pattern = "DetectedGazePattern"
    context = "Working"
    break_suggested = "No"
    detector = LoopDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        gaze_info["face_detected"] = len(faces) > 0
        gaze_info["eyes_detected"] = False
        gaze_info["blink_detected"] = False

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            gaze_info["eyes_detected"] = len(eyes) >= 2

            if len(eyes) == 0:
                if time.time() - self.last_blink_time > 0.5:
                    gaze_info["blink_detected"] = True
                    self.last_blink_time = time.time()

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 255), 2)
            break  # only first face

        duration = time.time() - start_time
        loop_type = detector.classify_loop(gaze_info, duration)
        break_suggested = "Yes" if loop_type in ["EscapeLoop", "FreezeLoop"] else "No"

        # 🧠 Professional Overlay Box
        overlay_text = [
            f"🌀 LoopType: {loop_type}",
            f"⏱ Duration: {int(duration)}s",
            f"🛑 Break: {break_suggested}",
            f"🧠 Pattern: {pattern}",
            f"🗂 Context: {context}",
        ]

        box_start_y = 30
        for i, line in enumerate(overlay_text):
            cv2.putText(
                frame,
                line,
                (10, box_start_y + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        cv2.rectangle(frame, (5, 5), (380, 200), (0, 0, 0), 2)
        cv2.imshow("📺 MirrorMind – Loop Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return gaze_info
