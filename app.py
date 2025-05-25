import streamlit as st
import pandas as pd
import os
import plotly.express as px
import clustering
import predictor
import passport_ocr
from datetime import datetime
import cv2
from deepface import DeepFace
import time
import subprocess
import uuid
st.set_page_config(page_title="AI Avtosalon Tahlili", layout="wide", page_icon="🚗")

st.markdown('''
<style>
html, body {
    font-family: 'Segoe UI', sans-serif;
    background-color: #f9f9f9;
}
.stSidebar {
    background-color: #f9f9f9;
    color: #000;
}
.stButton > button {
    background-color: #27ae60;
    color: white;
    border-radius: 5px;
}
</style>
''', unsafe_allow_html=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "customer_analysis_data")
CUSTOMERS_PATH = os.path.join(DATA_DIR, "customers.csv")
VISITS_PATH = os.path.join(DATA_DIR, "visits.csv")

def load_data():
    customers = pd.read_csv(CUSTOMERS_PATH)
    visits = pd.read_csv(VISITS_PATH)
    visits["Kirish_vaqti"] = pd.to_datetime(visits["Kirish_vaqti"])
    visits["Chiqish_vaqti"] = pd.to_datetime(visits["Chiqish_vaqti"])
    visits["Duration_min"] = (visits["Chiqish_vaqti"] - visits["Kirish_vaqti"]).dt.total_seconds() / 60
    return customers, visits

customers, visits = st.cache_data(load_data)()
predictor.train_model(CUSTOMERS_PATH, VISITS_PATH)

st.sidebar.title("🚘 Avtosalon AI Tahlili")
section = st.sidebar.radio("Bo'limni tanlang:", [
    "🏠 Dashboard",
    "🧠 AI Klasterlash",
    "🔮 Davomiylik Taxmini",
    "🎥 Kamera Tahlili",
    "🛰️ Yuqoridan Kuzatuv",
    "🆔 Pasport Ro'yxatdan O'tkazish"
])

if section == "🏠 Dashboard":
    st.title("🚗 Umumiy Tahlil")
    c1, c2, c3 = st.columns(3)
    c1.metric("Jami mijozlar", customers.shape[0])
    c2.metric("Jami tashriflar", visits.shape[0])
    c3.metric("O'rtacha davomiylik", f"{visits['Duration_min'].mean():.1f} daqiqa")

    st.markdown("---")
    tabs = st.tabs(["Jins", "Yosh", "Maqsad", "Soat", "Davomiylik"])

    with tabs[0]:
        st.subheader("👥 Jins bo'yicha taqsimot")
        gender_df = customers["Jins"].value_counts().rename_axis('Jins').reset_index(name='count')
        fig = px.pie(gender_df, names='Jins', values='count', color_discrete_sequence=["#301fb4", "#12f43b"])
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.subheader("🎂 Yosh bo'yicha taqsimot")
        fig = px.histogram(customers, x='Yosh', nbins=20, title='Yosh')
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.subheader("🎯 Tashrif maqsadi")
        purpose_df = visits['Maqsadi'].value_counts().rename_axis('Maqsad').reset_index(name='count')
        fig = px.bar(purpose_df, x='Maqsad', y='count', color='Maqsad', title='Maqsadlar', color_discrete_sequence=["#fff70e", '#2ca02c', "#841be6", "#ef3f1b", '#e377c2'])
        st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        st.subheader("🕒 Tashrif soati")
        visits['hour'] = visits['Kirish_vaqti'].dt.hour
        hour_df = visits['hour'].value_counts().sort_index().rename_axis('Soat').reset_index(name='count')
        fig = px.line(hour_df, x='Soat', y='count', markers=True, title="Soat bo'yicha tashriflar")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
        st.subheader("⏱️ Davomiylik")
        fig = px.histogram(visits, x='Duration_min', nbins=20, title='Tashrif davomiyligi (min)')
        st.plotly_chart(fig, use_container_width=True)

elif section == "🧠 AI Klasterlash":
    st.title("🧠 Mijozlar klasterlash")
    if st.button('🔍 Klasterlash'):
        dfc = clustering.run_clustering(CUSTOMERS_PATH, VISITS_PATH, n_clusters=4)
        st.dataframe(dfc, use_container_width=True)
        dist = dfc['Cluster_Label'].value_counts().rename_axis('Cluster').reset_index(name='count')
        fig = px.pie(dist, names='Cluster', values='count', color_discrete_sequence=["#e24b2d","#e82fb0","#30cffb","#3917cf"])
        st.plotly_chart(fig, use_container_width=True)

elif section == "🔮 Davomiylik Taxmini":
    st.title("🔮 Prediction of time user will spend")
    with st.form('form'):
        age = st.slider('Yosh', 18,80,30)
        gender = st.selectbox('Jins',['Erkak','Ayol'])
        purpose = st.selectbox('Maqsad', visits['Maqsadi'].unique())
        if st.form_submit_button('Taxminla'):
            pred = predictor.predict_duration(age, gender, purpose)
            st.success(f"Taxmin: {pred:.1f} daqiqa")

elif section == "🎥 Kamera Tahlili":
    st.title("🎥 Turniket camera preview")
    run = st.checkbox('Kamerani yoqish')
    frame_pl = st.empty()
    if run:
        cap = cv2.VideoCapture(0)
        last = 0
        res = {'age':None,'gender':None,'confidence':None,'time':None}
        while True:
            ret, img = cap.read()
            if not ret: break
            img = cv2.flip(img,1)
            now = time.time()
            if now-last>2:
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                faces = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml").detectMultiScale(gray,1.1,6)
                if len(faces):
                    x,y,w,h = max(faces,key=lambda f:f[2]*f[3])
                    face = img[y:y+h,x:x+w]
                    out = DeepFace.analyze(face,actions=['age','gender'],enforce_detection=False,silent=True)[0]
                    g = max(out['gender'],key=out['gender'].get)
                    res={'age':int(out['age']),'gender':g,'confidence':f"{out['gender'][g]:.1f}%",'time':datetime.now().strftime('%H:%M:%S')}
                last=now
            cv2.putText(img,f"{res['gender']}({res['confidence']})",(30,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
            cv2.putText(img,f"Yosh:{res['age']}",(30,60),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
            cv2.putText(img,f"{res['time']}",(30,90),cv2.FONT_HERSHEY_SIMPLEX,0.7,(200,200,0),1)
            frame_pl.image(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),use_container_width=True)
        cap.release()

elif section == "🛰️ Yuqoridan Kuzatuv":
    st.title("🛰️ Yuqoridan Kuzatuv Kamera (Zonalar)")
    run = st.checkbox("📡 Kamerani ishga tushurish")
    zone_display = st.empty()
    zone_log = st.empty()

    if run:
        cap = cv2.VideoCapture(0)  # Replace with your RTSP stream if needed
        W, H = 1280, 720
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

        zones = {
            "A": (100, 100, 400, 300),
            "B": (500, 100, 800, 300),
            "C": (100, 400, 400, 650),
            "D": (500, 400, 800, 650)
        }

        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        visit_log_path = VISITS_PATH
        if not os.path.exists(visit_log_path):
            pd.DataFrame(columns=["Tashrif_ID", "Pasport_raqami", "Kirish_vaqti", "Chiqish_vaqti", "Maqsadi", "Tracked"]).to_csv(visit_log_path, index=False)

        active_visits = {}

        def find_nearby_id(cx, cy):
            for pid, v in active_visits.items():
                px, py = v["center"]
                if abs(cx - px) < 60 and abs(cy - py) < 60:
                    return pid
            return None

        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Kamera oqimini olishda xatolik")
                break

            frame = cv2.resize(frame, (W, H))
            people, _ = hog.detectMultiScale(frame, winStride=(8, 8))
            now = datetime.now()

            seen_ids = set()

            for x, y, w, h in people:
                cx, cy = x + w // 2, y + h // 2
                current_zone = None
                for zone, (zx1, zy1, zx2, zy2) in zones.items():
                    if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                        current_zone = zone
                        break

                person_id = find_nearby_id(cx, cy)
                if not person_id:
                    person_id = f"TMP-{uuid.uuid4().hex[:8]}"
                    active_visits[person_id] = {
                        "center": (cx, cy),
                        "zone": current_zone,
                        "entry_time": now
                    }
                else:
                    active_visits[person_id]["center"] = (cx, cy)
                    if active_visits[person_id]["zone"] != current_zone:
                        row = {
                            "Tashrif_ID": int(time.time()),
                            "Pasport_raqami": person_id,
                            "Kirish_vaqti": active_visits[person_id]["entry_time"].strftime("%Y-%m-%d %H:%M:%S"),
                            "Chiqish_vaqti": now.strftime("%Y-%m-%d %H:%M:%S"),
                            "Maqsadi": f"Zone {active_visits[person_id]['zone']}",
                            "Tracked": "Yes"
                        }
                        df = pd.DataFrame([row])
                        df.to_csv(visit_log_path, mode='a', index=False, header=False)
                        active_visits[person_id]["zone"] = current_zone
                        active_visits[person_id]["entry_time"] = now

                seen_ids.add(person_id)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{current_zone or ''}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            active_visits = {k: v for k, v in active_visits.items() if k in seen_ids}

            for zone, (zx1, zy1, zx2, zy2) in zones.items():
                cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 0, 255), 2)
                cv2.putText(frame, f"Zone {zone}", (zx1 + 5, zy1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            zone_display.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            zone_log.markdown(f"⏱️ **Aktiv tashriflar**: {len(active_visits)}")

        cap.release()

elif section == "🆔 Pasport Ro'yxatdan O'tkazish":
    st.title("🆔 Pasport orqali ro'yxat")
    file = st.file_uploader('Rasm yuklang', type=['jpg','png','jpeg'])
    if file:
        path='temp.jpg'
        with open(path,'wb') as f: f.write(file.getbuffer())
        data=passport_ocr.extract_passport_data(path)
        st.json(data)
        if st.button('💾 Saqlash'):
            if st.session_state.get('save_customer') is None:
                pass
            if save_customer({
                'Pasport_raqami':data['Pasport_raqami'],'Ism':data['Ism'],'Jins':data['Jins'],'Yosh':data['Yosh'],'Telefon':'','Credit_Class':'C'
            }): st.success('Saqlandi')
            else: st.warning('Mavjud')
