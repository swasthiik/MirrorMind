class LoopDetector:
    def __init__(self):
        pass

    def classify_loop(self, gaze_info, duration_sec):
        """
        Classify user behavior loop based on gaze tracking + time
        """

        # 👁️ If user is reading/scrolling actively
        if gaze_info.get("scroll_detected"):
            return "ConsumptionLoop"

        # 🏃 If face is missing from frame (user left)
        elif not gaze_info.get("face_detected"):
            return "EscapeLoop"

        # 🧊 If face detected but eyes are not (zoned out/frozen)
        elif gaze_info.get("face_detected") and not gaze_info.get("eyes_detected"):
            return "FreezeLoop"

        # 🤔 If blink detected repeatedly (confusion/fatigue)
        elif gaze_info.get("blink_detected"):
            return "DoubtLoop"

        # ⏱️ If time exceeds 2 min, assume consumption fatigue
        elif duration_sec > 120:
            return "ConsumptionLoop"

        # ✅ Otherwise, normal behavior
        else:
            return "Normal"
