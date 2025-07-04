class LoopDetector:
    def __init__(self):
        pass

    def classify_loop(self, gaze_info, duration_sec):
        """
        Classify the loop type based on gaze_info and duration
        """

        # 😁 Smile = Comparison
        if gaze_info.get("smile_detected"):
            return "ComparisonLoop"

        # 📜 Scroll Activity = Passive Consumption
        if gaze_info.get("scroll_detected"):
            return "ConsumptionLoop"

        # 🙈 No face = Escape
        if not gaze_info.get("face_detected"):
            return "EscapeLoop"

        # 🧊 Face but no eyes = Freeze
        if gaze_info.get("face_detected") and not gaze_info.get("eyes_detected"):
            return "FreezeLoop"

        # 👀 Blink = Doubt
        if gaze_info.get("blink_detected"):
            return "DoubtLoop"

        # ⏱️ Long session = fatigue/consumption
        if duration_sec > 120:
            return "ConsumptionLoop"

        # ✅ Default
        return "Normal"
