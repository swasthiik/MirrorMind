# src/pattern_logger.py

import csv
from datetime import datetime

def log_pattern(pattern, loop_type, duration, context, break_suggested, file_path="data/loop_log_dataset.csv"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, pattern, loop_type, duration, context, break_suggested])
