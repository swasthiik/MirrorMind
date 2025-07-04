# ✅ app.py

import sys
import os
import streamlit as st
import pandas as pd
import time
from datetime import datetime
import altair as alt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.face_gaze_tracker import FaceGazeTracker
from src.loop_detector import LoopDetector
from src.pattern_logger import log_pattern

st.set_page_config(page_title="MirrorMind", layout="wide")
st.title("🧠 MirrorMind – Real-Time Loop Behavior Detection")

tab1, tab2 = st.tabs(["📸 Live Loop Tracker", "📊 Loop Log Viewer"])

with tab1:
    st.subheader("🛁 Real-Time Detection (Press 'Start')")

    if st.button("▶️ Start Detection"):
        tracker = FaceGazeTracker()
        detector = LoopDetector()

        frame_display = st.empty()
        loop_type = "Normal"
        pattern = "DetectedGazePattern"
        context = "Working"
        break_suggested = "No"

        start_time = time.time()

        for output in tracker.stream_gaze_overlay(
            loop_type=loop_type,
            pattern=pattern,
            context=context,
            break_suggested=break_suggested,
            max_duration=10
        ):
            frame_display.image(output["frame"], channels="RGB", caption="🯮 Robo Eye Active", use_column_width=True)

        duration = time.time() - start_time
        gaze_info = tracker.final_gaze_info

        loop_type = detector.classify_loop(gaze_info, duration)
        break_suggested = "Yes" if loop_type in ["EscapeLoop", "FreezeLoop"] else "No"

        log_pattern(
            pattern=pattern,
            loop_type=loop_type,
            duration=f"{int(duration)} sec",
            context=context,
            break_suggested=break_suggested,
            file_path="data/loop_log_dataset.csv"
        )

        st.success(f"🔁 Loop Detected: {loop_type}")
        st.write(f"📌 Pattern: {pattern}")
        st.write(f"⏱️ Duration: {int(duration)} sec")
        st.write(f"📌 Context: {context}")
        st.write(f"🚨 Break Suggested: {break_suggested}")

with tab2:
    st.subheader("📄 Loop Log Dataset")
    try:
        df = pd.read_csv("data/loop_log_dataset.csv")
        st.dataframe(df.tail(10), use_container_width=True)

        st.subheader("📊 Loop Type Distribution")
        loop_counts = df["LoopType"].value_counts()
        st.bar_chart(loop_counts)

        st.subheader("🚦 Break Suggestions Summary")
        break_df = df["BreakSuggested"].value_counts().reset_index()
        break_df.columns = ['BreakSuggested', 'Count']
        pie = alt.Chart(break_df).mark_arc().encode(
            theta=alt.Theta(field="Count", type="quantitative"),
            color=alt.Color(field="BreakSuggested", type="nominal"),
            tooltip=["BreakSuggested", "Count"]
        )
        st.altair_chart(pie, use_container_width=True)

        st.subheader("🧠 Recent Patterns")
        st.dataframe(df[["Pattern", "Context"]].tail(5), use_container_width=True)

    except FileNotFoundError:
        st.warning("⚠️ No dataset found yet. Run a session first.")
