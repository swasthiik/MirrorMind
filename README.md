<<<<<<< HEAD
# 🧠 MirrorMind – Real-Time Loop Detection Using Gaze & Expression

MirrorMind is a real-time behavior analysis system that tracks human facial expressions and gaze patterns to detect loop behaviors — such as zoning out, escaping reality, smiling, or freezing — while sitting in front of a screen. It leverages computer vision, face detection, and custom logic to classify these behaviors live.

## 🎥 Demo

Watch a real-time demo here: [https://www.linkedin.com/in/swasthiik](https://www.linkedin.com/in/swasthiik)

## 📌 Key Features

- 🧠 **Loop Detection Types**:
  - `Normal`: Stable presence and focus.
  - `FreezeLoop`: Stillness or lack of activity for extended time.
  - `EscapeLoop`: Sudden disappearance from the camera.
  - `ComparisonLoop`: Triggered by smiling (comparison thought).
  
- 👀 **Real-Time Gaze Tracking**
- 😊 **Smile Detection**
- ⚡ **Live Stream Behavior Updates** (no interruptions)
- 📈 **On-screen Visual Feedback** (loop state updates in real-time)
- 🧾 **Pattern Logging System** with timestamp, loop type, duration & context tag.

## 📂 Dataset

We built a 1000+ entry **human-labeled dataset** with fields like:
Timestamp	PatternDetected	LoopType	Duration	ContextTag	BreakSuggested
14:35:20	Looking away	EscapeLoop	00:10s	Overload	Yes

ruby
Copy
Edit

📎 Available here on Kaggle: [https://www.kaggle.com/datasets/swasthiik/mirrormind-loop-behavior-dataset](https://www.kaggle.com/datasets/swasthiik/mirrormind-loop-behavior-dataset)

## 🏗️ Project Structure

MirrorMind/
├── app/ # Streamlit frontend
│ └── app.py
├── src/
│ ├── face_gaze_tracker.py # Gaze & face detection
│ ├── loop_detector.py # Loop classification logic
│ └── pattern_logger.py # Logs user behavior
├── data/
│ └── loop_behavior_data.csv # Human-labeled dataset
├── requirements.txt
├── README.md

markdown
Copy
Edit

## ⚙️ Tech Stack

- Python 🐍
- OpenCV 🎥
- Streamlit 🌐
- Mediapipe (face mesh)
- NumPy & Pandas
- Real-time video processing

## 🚀 How to Run

```bash
# Install requirements
pip install -r requirements.txt

# Run the app
streamlit run app/app.py
⚠️ Webcam access is required for real-time tracking.

🔮 Future Improvements
🎯 Accuracy improvement for loop transitions

🧠 ML model training from labeled dataset

📊 Real-time chart visualizations

🌍 Multi-person detection support

🗣️ Emotion + voice tone analysis

🤝 About Me
Hi, I’m Swasthik – a passionate Machine Learning and Computer Vision enthusiast. MirrorMind is not just a project — it’s a reflection tool to help us become aware of our digital behaviors and emotional states.

📫 Connect with me:

🔗 LinkedIn: https://www.linkedin.com/in/swasthiik

💻 GitHub: https://github.com/swasthiik/swasthiik

📊 Kaggle: https://www.kaggle.com/swasthiik
=======
# MirrorMind
Real-Time Loop Behavior Detection using Gaze &amp; Facial Expression
>>>>>>> 17bdba9e1203bb85f33b92eec4d7dac47ed6551a
