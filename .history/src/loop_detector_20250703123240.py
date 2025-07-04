# ✅ loop_detector.py

class LoopDetector:
    def __init__(self):
        pass

    def classify_loop(self, gaze_info, duration_sec):
        if gaze_info.get("scroll_detected"):
            return "ConsumptionLoop"
        elif not gaze_info.get("face_detected"):
            return "EscapeLoop"
        elif gaze_info.get("face_detected") and not gaze_info.get("eyes_detected"):
            return "FreezeLoop"
        elif gaze_info.get("blink_detected"):
            return "DoubtLoop"
        elif duration_sec > 120:
            return "ConsumptionLoop"
        else:
            return "Normal"