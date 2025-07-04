def classify_loop(self, gaze_info, duration_sec):
    """
    Classify user behavior loop based on gaze + time
    """
    if not gaze_info["face_detected"]:
        return "EscapeLoop"
    
    elif gaze_info["blink_detected"]:
        return "DoubtLoop"

    elif gaze_info["face_detected"] and not gaze_info["eyes_detected"]:
        return "FreezeLoop"

    elif duration_sec > 120:
        return "ConsumptionLoop"

    else:
        return "Normal"
