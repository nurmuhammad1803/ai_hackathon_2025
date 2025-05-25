import streamlit as st
import cv2
import time
import numpy as np
from deepface import DeepFace
from datetime import datetime

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
ANALYSIS_INTERVAL = 2.0

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

st.set_page_config(layout="wide")
st.title("🧍 Kirish Kamerasi - Yuz, Yosh, Jins Aniqlash")
run_camera = st.checkbox("📷 Kamerani ishga tushurish")

frame_placeholder = st.empty()
prediction_placeholder = st.empty()

last_analysis = 0
latest_result = {
    "age": None,
    "gender": None,
    "confidence": None,
    "timestamp": None
}

if run_camera:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("❌ Kamera oqimi olinmadi")
            break

        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()

        current_time = time.time()
        if current_time - last_analysis > ANALYSIS_INTERVAL:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(150, 150))

            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
                face_img = frame[y:y+h, x:x+w]
                try:
                    result = DeepFace.analyze(face_img, actions=['age', 'gender'], enforce_detection=False, detector_backend='opencv', silent=True)[0]
                    gender = max(result['gender'], key=result['gender'].get)
                    conf = result['gender'][gender]

                    latest_result.update({
                        "age": int(result['age']),
                        "gender": gender,
                        "confidence": f"{conf:.1f}%",
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                except Exception as e:
                    st.error(f"Xatolik: {e}")
            last_analysis = current_time

        y0 = 40
        line_height = 30
        cv2.putText(display_frame, f"Jins: {latest_result['gender']} ({latest_result['confidence']})", (30, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Yosh: {latest_result['age']}", (30, y0 + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Vaqt: {latest_result['timestamp']}", (30, y0 + line_height*2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 1)

        frame_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        if not run_camera:
            break

    cap.release()
