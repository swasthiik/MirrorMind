# app/app.py

import streamlit as st
from src.face_gaze_tracker import FaceGazeTracker
from src.loop_detector import LoopDetector
from src.pattern_logger import log_pattern
import pandas as pd
import time
from datetime import datetime

st.set_page_config(page_title="MirrorMind", layout="wide")

st.title("🧠 MirrorMind – Real-Time Loop Behavior Detection")

tab1, tab2 = st.tabs(["📸 Live Loop Tracker", "📊 Loop Log Viewer"])

# --- TAB 1: REAL-TIME TRACKING ---
with tab1:
    st.subheader("📡 Real-Time Detection (Press 'Start')")

    if st.button("▶️ Start Detection"):
        tracker = FaceGazeTracker()
        detector = LoopDetector()

        start_time = time.time()
        gaze_info = tracker.track_gaze()  # OpenCV will open webcam
        duration = time.time() - start_time

        loop_type = detector.classify_loop(gaze_info, duration)
        pattern = "DetectedGazePattern"
        context = "Working"  # Can be dynamic later
        break_suggested = "Yes" if loop_type in ["EscapeLoop", "FreezeLoop"] else "No"

        # Logging to dataset
        log_pattern(
            pattern, loop_type, f"{int(duration)} sec", context, break_suggested,
            file_path="data/MirrorMind_LoopBehaviorDataset.csv"
        )

        st.success(f"🔁 Loop Detected: {loop_type}")
        st.write(f"📌 Pattern: {pattern}")
        st.write(f"⏱️ Duration: {int(duration)} sec")
        st.write(f"📎 Context: {context}")
        st.write(f"🚨 Break Suggested: {break_suggested}")

# --- TAB 2: DATASET VIEW + STATS ---
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
