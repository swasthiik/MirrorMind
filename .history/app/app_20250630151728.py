# app/app.py

import sys
import os
# 🔧 Add src/ folder to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from src.face_gaze_tracker import FaceGazeTracker
from src.loop_detector import LoopDetector
from src.pattern_logger import log_pattern
import pandas as pd
import time
from datetime import datetime

# --- Streamlit Page Setup ---
st.set_page_config(page_title="MirrorMind", layout="wide")
st.title("🧠 MirrorMind – Real-Time Loop Behavior Detection")

# --- Create Tabs ---
tab1, tab2 = st.tabs(["📸 Live Loop Tracker", "📊 Loop Log Viewer"])

# ===============================
# 📍 TAB 1: LIVE LOOP TRACKER
# ===============================
with tab1:
    st.subheader("📡 Real-Time Detection (Press 'Start')")

    if st.button("▶️ Start Detection"):
        tracker = FaceGazeTracker()
        detector = LoopDetector()

        start_time = time.time()
        gaze_info = tracker.track_gaze()  # Webcam will open
        duration = time.time() - start_time

        loop_type = detector.classify_loop(gaze_info, duration)
        pattern = "DetectedGazePattern"
        context = "Working"
        break_suggested = "Yes" if loop_type in ["EscapeLoop", "FreezeLoop"] else "No"

        # 🔁 Log the pattern
        log_pattern(
            pattern=pattern,
            loop_type=loop_type,
            duration=f"{int(duration)} sec",
            context=context,
            break_suggested=break_suggested,
            file_path="data/MirrorMind_LoopBehaviorDataset.csv"
        )

        # 🟢 Output to user
        st.success(f"🔁 Loop Detected: {loop_type}")
        st.write(f"📌 Pattern: {pattern}")
        st.write(f"⏱️ Duration: {int(duration)} sec")
        st.write(f"📎 Context: {context}")
        st.write(f"🚨 Break Suggested: {break_suggested}")

# ===============================
# 📍 TAB 2: DATASET & CHARTS
# ===============================
with tab2:
    st.subheader("📄 Loop Log Dataset")

    try:
        df = pd.read_csv("data/MirrorMind_LoopBehaviorDataset.csv")
        st.dataframe(df.tail(10), use_container_width=True)

        st.subheader("📊 Loop Type Distribution")
        loop_counts = df["LoopType"].value_counts()
        st.bar_chart(loop_counts)

        st.subheader("🚦 Break Suggestions Summary")
        break_chart = df["BreakSuggested"].value_counts()
        st.pie_chart(break_chart)

    except FileNotFoundError:
        st.warning("⚠️ No dataset found yet. Run a session first.")
