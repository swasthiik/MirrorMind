# ✅ app.py
import sys
import os
import streamlit as st
import pandas as pd
import time
import altair as alt

# Add src path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.face_gaze_tracker import FaceGazeTracker
from src.loop_detector import LoopDetector
from src.pattern_logger import log_pattern

# Streamlit config
st.set_page_config(page_title="MirrorMind", layout="wide")
st.title("🧠 MirrorMind – Real-Time Loop Behavior Detection")

# Tabs
tab1, tab2 = st.tabs(["📸 Live Loop Tracker", "📊 Loop Log Viewer"])

# ========================
# 🔴 LIVE LOOP DETECTION
# ========================
with tab1:
    st.subheader("🛁 Real-Time Detection (Press 'Start')")
    st.markdown("🧪 Press 'q' in the live window to stop detection")

    if st.button("▶️ Start Detection"):
        tracker = FaceGazeTracker()
        detector = LoopDetector()

        pattern = "DetectedGazePattern"
        start_time = time.time()

        # ✅ Start live stream (no loop here)
        tracker.stream_gaze_overlay_live()

        # After stream ends (user presses 'q')
        duration = time.time() - start_time
        gaze_info = tracker.final_gaze_info
        context = gaze_info.get("context", "AutoDetect")

        loop_type = detector.classify_loop(gaze_info, duration)
        if gaze_info.get("smile_detected"):
            loop_type = "ComparisonLoop"

        break_suggested = "Yes" if loop_type in ["EscapeLoop", "FreezeLoop"] else "No"

        log_pattern(
            pattern=pattern,
            loop_type=loop_type,
            duration_sec=duration,
            context=context,
            break_suggested=break_suggested,
            file_path="data/loop_log_dataset.csv"
        )

        st.success(f"🔁 Loop Detected: {loop_type}")
        st.write(f"📌 Pattern: {pattern}")
        st.write(f"⏱️ Duration: {int(duration)} sec")
        st.write(f"📎 Context: {context}")
        st.write(f"🚨 Break Suggested: {break_suggested}")

# ========================
# 📊 LOG VIEWER
# ========================
with tab2:
    st.subheader("📄 Loop Log Dataset")

    try:
        df = pd.read_csv("data/loop_log_dataset.csv")
        st.dataframe(df.tail(10), use_container_width=True)

        st.subheader("📊 Loop Type Distribution")
        st.bar_chart(df["LoopType"].value_counts())

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
        if "Pattern" in df.columns and "Context" in df.columns:
            st.dataframe(df[["Pattern", "Context"]].tail(5), use_container_width=True)
        else:
            st.warning("🛑 Columns 'Pattern' or 'Context' missing in CSV")

    except FileNotFoundError:
        st.warning("⚠️ No dataset found yet. Run a session first.")
