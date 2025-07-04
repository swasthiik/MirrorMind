class LoopDetector:
    def __init__(self):
        pass

    def classify_loop(self, gaze_info, duration_sec):
        if gaze_info.get("scroll_detected"):
            return "ScrollLoop"
        elif not gaze_info["face_detected"]:
            return "EscapeLoop"
        elif gaze_info["face_detected"] and not gaze_info["eyes_detected"]:
            return "FreezeLoop"
        elif gaze_info["blink_detected"]:
            return "DoubtLoop"
        elif duration_sec > 120:
            return "ConsumptionLoop"
        else:
            return "Normal"
