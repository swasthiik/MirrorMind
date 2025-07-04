# app/app.py

from src.face_gaze_tracker import FaceGazeTracker
from src.loop_detector import LoopDetector
from src.pattern_logger import log_pattern
import time

tracker = FaceGazeTracker()
detector = LoopDetector()

print("👁️ MirrorMind is starting... (press Q to quit webcam)")

start_time = time.time()
gaze_info = tracker.track_gaze()
duration = time.time() - start_time

loop_type = detector.classify_loop(gaze_info, duration)
pattern = "DetectedGazePattern"
context = "Work"  # Or can be auto-inferred in advanced version
break_suggested = "Yes" if loop_type in ["EscapeLoop", "FreezeLoop"] else "No"

log_pattern(pattern, loop_type, f"{int(duration)} sec", context, break_suggested)

print(f"\n🔍 Pattern: {pattern}")
print(f"🔁 LoopType: {loop_type}")
print(f"🧠 Break Suggested: {break_suggested}")
