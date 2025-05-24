import cv2
from deepface import DeepFace
from datetime import datetime, timedelta
import threading
import queue
import time
import numpy as np
import logging
import pandas as pd
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG = {
    "cascade_path": cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
    "analysis_interval": 2.0,
    "frame_size": (1280, 720),
    "min_face_size": (200, 200),
    "font": cv2.FONT_HERSHEY_COMPLEX,
    "text_scale": 1.0,
    "text_thickness": 2,
    "mirror": True
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "customer_analysis_data")
CUSTOMERS_PATH = os.path.join(DATA_DIR, "customers.csv")
VISITS_PATH = os.path.join(DATA_DIR, "visits.csv")

class FaceAnalyzer:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(CONFIG["cascade_path"])
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar Cascade")

        self.analysis_queue = queue.Queue(maxsize=1)
        self.latest_result = {
            "age": None,
            "gender": None,
            "gender_confidence": None,
            "timestamp": None,
            "processing": False,
            "face_detected": False
        }

        self.recent_entries = [] 

        os.makedirs(DATA_DIR, exist_ok=True)
        self._init_data_files()
        self._init_camera()
        self._start_worker()

    def _init_data_files(self):
        if not os.path.exists(CUSTOMERS_PATH):
            pd.DataFrame(columns=["Pasport_raqami", "Ism", "Jins", "Yosh", "Telefon", "Credit_Class"]).to_csv(CUSTOMERS_PATH, index=False)
        if not os.path.exists(VISITS_PATH):
            pd.DataFrame(columns=["Tashrif_ID", "Pasport_raqami", "Kirish_vaqti", "Chiqish_vaqti", "Maqsadi", "Tracked"]).to_csv(VISITS_PATH, index=False)

    def _init_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot access camera")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["frame_size"][0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["frame_size"][1])
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 140)

    def _start_worker(self):
        threading.Thread(target=self._analyze_faces, daemon=True).start()

    def _analyze_faces(self):
        while True:
            frame = self.analysis_queue.get()
            self.latest_result["processing"] = True

            try:
                result = self._process_frame(frame)
                self.latest_result.update(result)
                if result["face_detected"]:
                    self._log_visit(result)
            except Exception as e:
                logger.error(f"Analysis error: {str(e)}")
                self.latest_result.update({
                    "age": None,
                    "gender": "Analysis error",
                    "gender_confidence": "0%",
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "processing": False
                })

    def _process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=CONFIG["min_face_size"])

        if len(faces) == 0:
            return {
                "age": None,
                "gender": "No face detected",
                "gender_confidence": "0%",
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "processing": False,
                "face_detected": False
            }

        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
        face_img = frame[y:y+h, x:x+w]

        face_img = cv2.bilateralFilter(face_img, 11, 75, 75)
        face_img = cv2.normalize(face_img, None, 0, 255, cv2.NORM_MINMAX)

        analysis = DeepFace.analyze(face_img, actions=['age', 'gender'], enforce_detection=False, detector_backend='opencv', align=True, silent=True)[0]
        female_conf = analysis['gender']['Woman']
        male_conf = analysis['gender']['Man']

        gender = "Female" if female_conf > male_conf else "Male"
        confidence = max(female_conf, male_conf)

        return {
            "age": max(0, int(analysis['age']) - 4),
            "gender": gender,
            "gender_confidence": f"{confidence:.1f}%",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "processing": False,
            "face_detected": True
        }

    def _log_visit(self, result):
        now = datetime.now()
        age = result["age"]
        gender = result["gender"]

        for entry in self.recent_entries:
            if abs(entry[0] - age) <= 2 and entry[1] == gender:
                if (now - entry[2]).seconds < 600:
                    return

        self.recent_entries.append((age, gender, now))
        self.recent_entries = [e for e in self.recent_entries if (now - e[2]).seconds < 600]

        passport_id = f"TMP-{now.strftime('%Y%m%d%H%M%S')}"
        customer_df = pd.read_csv(CUSTOMERS_PATH)
        customer_df = pd.concat([customer_df, pd.DataFrame([{ "Pasport_raqami": passport_id, "Ism": "Noma'lum", "Jins": gender, "Yosh": age, "Telefon": "", "Credit_Class": "C" }])])
        customer_df.to_csv(CUSTOMERS_PATH, index=False)

        visits_df = pd.read_csv(VISITS_PATH)
        new_visit = {
            "Tashrif_ID": len(visits_df),
            "Pasport_raqami": passport_id,
            "Kirish_vaqti": now.strftime("%Y-%m-%d %H:%M:%S"),
            "Chiqish_vaqti": now.strftime("%Y-%m-%d %H:%M:%S"),
            "Maqsadi": "Noma'lum",
            "Tracked": "No"
        }
        visits_df = pd.concat([visits_df, pd.DataFrame([new_visit])])
        visits_df.to_csv(VISITS_PATH, index=False)
        logger.info(f"Logged new visit: {passport_id} - {gender}, {age} y.o.")

    def _display_info(self, frame):
        y_offset = 50
        line_height = 40

        if self.latest_result["gender"]:
            gender_text = f"{self.latest_result['gender']} ({self.latest_result['gender_confidence']})"
            cv2.putText(frame, gender_text, (30, y_offset), CONFIG["font"], CONFIG["text_scale"], (0, 255, 0), CONFIG["text_thickness"])
            y_offset += line_height

            if self.latest_result["age"] is not None:
                age_text = f"Age: {self.latest_result['age']}"
                cv2.putText(frame, age_text, (30, y_offset), CONFIG["font"], CONFIG["text_scale"], (0, 255, 0), CONFIG["text_thickness"])
                y_offset += line_height

            time_text = f"Time: {self.latest_result['timestamp']}"
            cv2.putText(frame, time_text, (30, y_offset), CONFIG["font"], CONFIG["text_scale"] * 0.8, (200, 200, 0), CONFIG["text_thickness"] - 1)
            y_offset += line_height

        status_text = "Analyzing..." if self.latest_result["processing"] else "Ready"
        status_color = (0, 0, 255) if self.latest_result["processing"] else (0, 255, 0)
        cv2.putText(frame, status_text, (30, y_offset), CONFIG["font"], CONFIG["text_scale"] * 0.8, status_color, CONFIG["text_thickness"] - 1)
        cv2.putText(frame, "Press 'q' to exit", (30, CONFIG["frame_size"][1] - 30), CONFIG["font"], 0.7, (255, 255, 255), 1)

    def run(self):
        last_analysis_time = 0
        cv2.namedWindow("Face Analyzer", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Face Analyzer", *CONFIG["frame_size"])

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Frame read error")
                    break

                if CONFIG["mirror"]:
                    frame = cv2.flip(frame, 1)

                current_time = time.time()
                if (current_time - last_analysis_time > CONFIG["analysis_interval"] and 
                    not self.latest_result["processing"] and 
                    self.analysis_queue.empty()):
                    self.analysis_queue.put(frame.copy())
                    last_analysis_time = current_time

                self._display_info(frame)

                cv2.imshow("Face Analyzer", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting face analyzer...")
    analyzer = FaceAnalyzer()
    analyzer.run()
