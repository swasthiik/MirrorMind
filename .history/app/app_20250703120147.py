# ✅ app/app.py

import sys
import os
import streamlit as st
import pandas as pd
import time
from datetime import datetime
import altair as alt

# ✅ Add src path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.face_gaze_tracker import FaceGazeTracker
from src.loop_detector import LoopDetector
from src.pattern_logger import log_pattern

# 🎨 Streamlit Config
st.set_page_config(page_title="MirrorMind", layout="wide")
st.title("🧠 MirrorMind – Real-Time Loop Behavior Detection")

# 🚀 Tabs
tab1, tab2 = st.tabs(["📸 Live Loop Tracker", "📊 Loop Log Viewer"])

# ================================
# 📍 TAB 1: Real-Time Detection
# ================================
with tab1:
    st.subheader("📡 Real-Time Detection (Press 'Start')")

    if st.button("▶️ Start Detection"):
        tracker = FaceGazeTracker()
        detector = LoopDetector()

        # Placeholder for video frame
        frame_display = st.image([], caption="🦾 Robo Eye Active", use_column_width=True)

        # Start timer
        start_time = time.time()
        gaze_data = []

        # Dummy initial loop info (later classify after session ends)
        loop_type = "Processing..."
        pattern = "DetectedGazePattern"
        context = "Working"
        break_suggested = "No"

        # Start webcam stream for 10 seconds
        for frame in tracker.stream_gaze_overlay(
            loop_type=loop_type,
            pattern=pattern,
            context=context,
            break_suggested=break_suggested,
            max_duration=10
        ):
            frame_display.image(frame, channels="RGB")

        # Calculate final duration
        duration = time.time() - start_time
        gaze_info = {"gaze_data": "some_data"}  # Minimal placeholder for loop_detector

        # 🔁 Loop classification
        loop_type = detector.classify_loop(gaze_info, duration)
        break_suggested = "Yes" if loop_type in ["EscapeLoop", "FreezeLoop"] else "No"

        # 📝 Log the result
        log_pattern(
            pattern=pattern,
            loop_type=loop_type,
            duration=f"{int(duration)} sec",
            context=context,
            break_suggested=break_suggested,
            file_path="data/loop_log_dataset.csv"
        )

        # ✅ Output Summary
        st.success(f"🔁 Loop Detected: {loop_type}")
        st.write(f"📌 Pattern: {pattern}")
        st.write(f"⏱️ Duration: {int(duration)} sec")
        st.write(f"📎 Context: {context}")
        st.write(f"🚨 Break Suggested: {break_suggested}")

# ================================
# 📊 TAB 2: Dataset + Chart View
# ================================
with tab2:
    st.subheader("📄 Loop Log Dataset")

    try:
        df = pd.read_csv("data/loop_log_dataset.csv")
        st.dataframe(df.tail(10), use_container_width=True)

        # 📊 Loop Type Distribution
        st.subheader("📊 Loop Type Distribution")
        loop_counts = df["LoopType"].value_counts()
        st.bar_chart(loop_counts)

        # 🚦 Break Suggestion Pie Chart
        st.subheader("🚦 Break Suggestions Summary")
        break_df = df["BreakSuggested"].value_counts().reset_index()
        break_df.columns = ['BreakSuggested', 'Count']
        pie = alt.Chart(break_df).mark_arc().encode(
            theta=alt.Theta(field="Count", type="quantitative"),
            color=alt.Color(field="BreakSuggested", type="nominal"),
            tooltip=["BreakSuggested", "Count"]
        )
        st.altair_chart(pie, use_container_width=True)

        # 🧠 Pattern + Context
        st.subheader("🧠 Recent Patterns")
        st.dataframe(df[["Pattern", "Context"]].tail(5), use_container_width=True)

    except FileNotFoundError:
        st.warning("⚠️ No dataset found yet. Run a session first.")
