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

# Page configuration
st.set_page_config(page_title="AI Avtosalon Tahlili", layout="wide", page_icon="🚗")

# Custom CSS for fonts and colors
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

# Data paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "customer_analysis_data")
CUSTOMERS_PATH = os.path.join(DATA_DIR, "customers.csv")
VISITS_PATH = os.path.join(DATA_DIR, "visits.csv")

# Load data with caching
def load_data():
    customers = pd.read_csv(CUSTOMERS_PATH)
    visits = pd.read_csv(VISITS_PATH)
    visits["Kirish_vaqti"] = pd.to_datetime(visits["Kirish_vaqti"])
    visits["Chiqish_vaqti"] = pd.to_datetime(visits["Chiqish_vaqti"])
    visits["Duration_min"] = (visits["Chiqish_vaqti"] - visits["Kirish_vaqti"]).dt.total_seconds() / 60
    return customers, visits

customers, visits = st.cache_data(load_data)()
# Train prediction model
predictor.train_model(CUSTOMERS_PATH, VISITS_PATH)

# Sidebar navigation
st.sidebar.title("🚘 Avtosalon AI Tahlili")
section = st.sidebar.radio("Bo'limni tanlang:", [
    "🏠 Dashboard",
    "🧠 AI Klasterlash",
    "🔮 Davomiylik Taxmini",
    "🎥 Kamera Tahlili",
    "🆔 Pasport Ro'yxatdan O'tkazish"
])

# Dashboard
if section == "🏠 Dashboard":
    st.title("🚗 Umumiy Tahlil")
    c1, c2, c3 = st.columns(3)
    c1.metric("Jami mijozlar", customers.shape[0])
    c2.metric("Jami tashriflar", visits.shape[0])
    c3.metric("O'rtacha davomiylik", f"{visits['Duration_min'].mean():.1f} daqiqa")

    st.markdown("---")
    tabs = st.tabs(["Jins", "Yosh", "Maqsad", "Soat", "Davomiylik"])

    # Jins distribution
    with tabs[0]:
        st.subheader("👥 Jins bo'yicha taqsimot")
        gender_df = customers["Jins"].value_counts().rename_axis('Jins').reset_index(name='count')
        fig = px.pie(gender_df, names='Jins', values='count', color_discrete_sequence=["#301fb4", "#12f43b"])
        st.plotly_chart(fig, use_container_width=True)

    # Yosh distribution
    with tabs[1]:
        st.subheader("🎂 Yosh bo'yicha taqsimot")
        fig = px.histogram(customers, x='Yosh', nbins=20, title='Yosh')
        st.plotly_chart(fig, use_container_width=True)

    # Maqsad
    with tabs[2]:
        st.subheader("🎯 Tashrif maqsadi")
        purpose_df = visits['Maqsadi'].value_counts().rename_axis('Maqsad').reset_index(name='count')
        fig = px.bar(purpose_df, x='Maqsad', y='count', color='Maqsad', title='Maqsadlar', color_discrete_sequence=["#fff70e", '#2ca02c', "#841be6", "#ef3f1b", '#e377c2'])
        st.plotly_chart(fig, use_container_width=True)

    # Soat
    with tabs[3]:
        st.subheader("🕒 Tashrif soati")
        visits['hour'] = visits['Kirish_vaqti'].dt.hour
        hour_df = visits['hour'].value_counts().sort_index().rename_axis('Soat').reset_index(name='count')
        fig = px.line(hour_df, x='Soat', y='count', markers=True, title="Soat bo'yicha tashriflar")
        st.plotly_chart(fig, use_container_width=True)

    # Davomiylik
    with tabs[4]:
        st.subheader("⏱️ Davomiylik")
        fig = px.histogram(visits, x='Duration_min', nbins=20, title='Tashrif davomiyligi (min)')
        st.plotly_chart(fig, use_container_width=True)

# AI Klasterlash
elif section == "🧠 AI Klasterlash":
    st.title("🧠 Mijozlar klasterlash")
    if st.button('🔍 Klasterlash'):
        dfc = clustering.run_clustering(CUSTOMERS_PATH, VISITS_PATH, n_clusters=4)
        st.dataframe(dfc, use_container_width=True)
        dist = dfc['Cluster_Label'].value_counts().rename_axis('Cluster').reset_index(name='count')
        fig = px.pie(dist, names='Cluster', values='count', color_discrete_sequence=["#e24b2d","#e82fb0","#30cffb","#3917cf"])
        st.plotly_chart(fig, use_container_width=True)

# Davomiylik Taxmini
elif section == "🔮 Davomiylik Taxmini":
    st.title("🔮 Davomiylikni taxmin qilish")
    with st.form('form'):
        age = st.slider('Yosh', 18,80,30)
        gender = st.selectbox('Jins',['Erkak','Ayol'])
        purpose = st.selectbox('Maqsad', visits['Maqsadi'].unique())
        if st.form_submit_button('Taxminla'):
            pred = predictor.predict_duration(age, gender, purpose)
            st.success(f"Taxmin: {pred:.1f} daqiqa")

# Kamera Tahlili
elif section == "🎥 Kamera Tahlili":
    st.title("🎥 Kirish kamerasidan tahlil")
    run = st.checkbox('📷 Ishga tushur')
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

# Pasport Ro'yxatdan O'tkazish
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
