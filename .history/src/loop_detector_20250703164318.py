class LoopDetector:
    def __init__(self):
        pass

    def classify_loop(self, gaze_info, duration_sec):
        if gaze_info.get("scroll_detected"):
            return "ConsumptionLoop"
        elif not gaze_info.get("face_detected"):
            return "EscapeLoop"  # ✅ No face in frame
        elif gaze_info.get("face_detected") and not gaze_info.get("eyes_detected"):
            return "FreezeLoop"
        elif gaze_info.get("blink_detected"):
            return "DoubtLoop"
        elif gaze_info.get("smile_detected"):
            return "ComparisonLoop"
        elif duration_sec > 120:
            return "ConsumptionLoop"
        else:
            return "Normal"
