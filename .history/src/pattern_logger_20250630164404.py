import pandas as pd
import os
from datetime import datetime

def log_pattern(pattern, loop_type, duration, context, break_suggested, file_path, scroll_detected=None):
    new_row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Pattern": pattern,
        "LoopType": loop_type,
        "Duration": duration,
        "Context": context,
        "BreakSuggested": break_suggested
    }

    if scroll_detected is not None:
        new_row["ScrollDetected"] = scroll_detected

    print("🚨 Logging row:", new_row)  # ✅ DEBUG line

    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            df = pd.DataFrame([new_row])

        df.to_csv(file_path, index=False)
        print(f"✅ Log successful: {file_path}")  # ✅ DEBUG line
    except Exception as e:
        print("❌ Log failed:", e)
