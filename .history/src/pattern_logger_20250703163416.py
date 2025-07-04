import pandas as pd
import os
from datetime import datetime

def log_pattern(pattern, loop_type, duration_sec, context, break_suggested, file_path):
    new_row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "PatternDetected": pattern,
        "LoopType": loop_type,
        "Duration": f"{round(duration_sec / 60, 1)} min",  # convert sec to min
        "ContextTag": context,
        "BreakSuggested": break_suggested
    }

    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            df = pd.DataFrame([new_row])

        df.to_csv(file_path, index=False)
        print(f"✅ Logged: {new_row}")
    except Exception as e:
        print("❌ Logging failed:", e)
